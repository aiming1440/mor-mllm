# llava_phi_mor.py

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .phi.configuration_phi import PhiConfig
from .phi.modeling_phi import PhiModel, PhiForCausalLM, PhiDecoderLayer
from transformers.models.phi.modeling_phi import PhiModel, PhiForCausalLM, PhiRotaryEmbedding

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.utils import logging

# --- MoR & KV Cache Imports ---
# Assumes the relevant MoR repo files have already been placed in the appropriate location
# e.g.: moellava/model/language_model/mor/
from ..mor_modules.expert_choice_router import MoRPhiDecoderLayer as MoRExpertChoiceDecoderLayer
from ..mor_modules.token_choice_router import MoRPhiTokenChoiceDecoderLayer as MoRTokenChoiceDecoderLayer
from ..mor_modules.cache_utils import Cache, DynamicCache, RecursiveDynamicCache

logger = logging.get_logger(__name__)

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def build_shared_recursion_block(source_layers, num_recursions, base_depth, mor_block_init):
    """Build the MoR shared submodule M(.;Theta).

    Args:
        source_layers: the original layer sequence to be collapsed (length must be
            num_recursions * base_depth); for the "cycle" strategy this is all layers,
            for the "middle_cycle" strategy this is the middle layers after trimming
            the first and last.
        mor_block_init: "grouped_sharing" (uses the original weights of
            the first group as the initial value for M) or "mean" (uses the elementwise
            mean of the corresponding-position layer weights across all groups).

    Returns:
        An nn.ModuleList of length base_depth representing the shared submodule M.
    """
    groups = [
        list(source_layers[r_idx * base_depth: (r_idx + 1) * base_depth])
        for r_idx in range(num_recursions)
    ]

    if mor_block_init == "grouped_sharing":
        # Selection-based: directly reuse the first group's original layer objects/weights as the shared submodule.
        canonical_block = nn.ModuleList(groups[0])
    elif mor_block_init == "mean":
        import copy
        canonical_block = nn.ModuleList([copy.deepcopy(layer) for layer in groups[0]])
        for depth_idx in range(base_depth):
            layers_at_this_depth = [groups[r_idx][depth_idx] for r_idx in range(num_recursions)]
            state_dicts = [layer.state_dict() for layer in layers_at_this_depth]
            averaged_state = {}
            for key in state_dicts[0].keys():
                stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
                averaged_state[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)
            canonical_block[depth_idx].load_state_dict(averaged_state)
    else:
        raise ValueError(f"Unknown mor_block_init strategy: {mor_block_init}")

    return canonical_block
# ==========================================================================================
# 1. Define MoR-related config and output classes
#    This part is adapted from the MoR repo, used to define MoR hyperparameters and the
#    unified output format
# ==========================================================================================

class MoRLLaVAPhiConfig(PhiConfig):
    model_type = "mor_llava_phi"

    def __init__(self,
                 mor_enable=True,
                 mor_mode="expert",
                 sharing_strategy="middle_cycle",
                 num_recursions=3,
                 group_size=None,
                 mor_block_init="grouped_sharing",
                 routing_threshold=0.8,
                 entropy_reg_coeff=0.1,
                 router=dict(rand_router=False, router_type="linear"),
                 gating="weighted",
                 expert=dict(cap_warmup_step=None, expert_capacity="0.5, 0.3, 0.2"),
                 token=dict(balancing="loss", router_func="softmax", alpha=1.0, bal_warmup_step=0),
                 kv_sharing=dict(enable=True, update_cache=False),
                 use_aux_loss=True,
                 aux_loss_coeff=0.1,
                 bal_loss_coeff=0.1,
                 use_z_loss=True,
                 z_loss_coeff=0.1,
                 **kwargs):
        """MoR-MLLM config.

        ``num_recursions = (num_hidden_layers - 2) // group_size`` (n=32, m=5 -> k=6).

        ``routing_threshold`` is functionally equivalent to the existing implementation's
        ``expert.expert_capacity`` (top-k capacity factor): capacity_factor ~= 1 - beta.
        The concrete capacity-factor-based routing implementation is kept as-is; 

        - "grouped_sharing" (default): every MoR block position reuses the same "canonical" original layer weights (the same physical parameters are
          shared/reused by all num_recursions positions).
        - "mean": constructs one shared parameter set from the mean of the
          corresponding-position layer weights across the num_recursions groups.
        The key common point between the two is that different depth positions ultimately
        share the same physical parameters (this is the core difference between "Mixture
        of Recursions" and ordinary layer-wise MoD); only the source of the initial value
        differs.
        """

        # Wrap all MoR-related config in a single dict to keep things tidy
        self.mor = dict(
            mor_enable=mor_enable,
            mor_mode=mor_mode,  # "expert" or "token"
            sharing_strategy=sharing_strategy,  # "cycle" or "middle_cycle"
            num_recursions=num_recursions,
            group_size=group_size,  # m, only used to derive the default of num_recursions
            mor_block_init=mor_block_init,  # "grouped_sharing" or "mean"
            routing_threshold=routing_threshold,  # beta, alignment metadata
            entropy_reg_coeff=entropy_reg_coeff,  # lambda_ent, default 0.1
            router=router,  # Router config
            expert=expert,  # For expert-choice, can be a comma-separated string for different
            token=token,  # For token-choice: balancing mode, router activation, alpha, warmup
            kv_sharing=kv_sharing,  # KV cache sharing config
            use_aux_loss=use_aux_loss,
            aux_loss_coeff=aux_loss_coeff,
            bal_loss_coeff=bal_loss_coeff,
            use_z_loss=use_z_loss,
            z_loss_coeff=z_loss_coeff,
            gating=gating,  # "hard" or "weighted"
            train_modules=[
                # 'router'
            ]
        )

        super().__init__(**kwargs)

@dataclass
class MoRBaseModelOutputWithPast(ModelOutput):
    """
    MoR model base output class, including detailed diagnostic metrics.
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    # --- Expert-choice routing metrics ---
    # Auxiliary router's binary cross-entropy loss (used to predict whether a token is selected)
    expert_aux_loss: Optional[torch.FloatTensor] = None
    # Auxiliary router's prediction accuracy
    expert_aux_acc: Optional[torch.FloatTensor] = None
    # Ratio of "dead" tokens (tokens never selected at the last recursion step)
    expert_dead_token_ratio: Optional[torch.FloatTensor] = None

    # --- Token-choice routing metrics ---
    # Load-balancing loss
    token_balance_loss: Optional[torch.FloatTensor] = None
    # Metric measuring the degree of load imbalance (Max Violation)
    token_max_violation: Optional[torch.FloatTensor] = None
    # Entropy measuring the diversity of router assignments
    token_router_entropy: Optional[torch.FloatTensor] = None

    # --- Generic routing metrics ---
    # Regularization loss used to penalize overly large logits
    router_z_loss: Optional[torch.FloatTensor] = None
    # Entropy regularization loss: computes the entropy of the routing
    # selection probability at each MoR block position; higher is better (encourages
    # diversity in the recursion-depth distribution across tokens, preventing collapse
    # into "all tokens have the same depth"). During training, its negative is multiplied
    # by entropy_reg_coeff and added to the total loss (i.e. minimizing negative entropy =
    # maximizing entropy).
    entropy_loss: Optional[torch.FloatTensor] = None
    # Sum of all (weighted) auxiliary losses, for convenient addition to the main loss
    total_aux_loss: Optional[torch.FloatTensor] = None

@dataclass
class MoRCausalLMOutputWithPast(ModelOutput):
    """
    MoR causal language model output class.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # Mirror all diagnostic fields from MoRBaseModelOutputWithPast here
    expert_aux_loss: Optional[torch.FloatTensor] = None
    expert_aux_acc: Optional[torch.FloatTensor] = None
    expert_dead_token_ratio: Optional[torch.FloatTensor] = None
    token_balance_loss: Optional[torch.FloatTensor] = None
    token_max_violation: Optional[torch.FloatTensor] = None
    token_router_entropy: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None
    entropy_loss: Optional[torch.FloatTensor] = None
    total_aux_loss: Optional[torch.FloatTensor] = None


# ==========================================================================================
# 2. Implement the MoR version of PhiModel
#    This is the core of the modification, rewriting the forward loop to support
#    recursion and routing
# ==========================================================================================

class MoRLLaVAPhiModel(LlavaMetaModel, PhiModel):
    config_class = MoRLLaVAPhiConfig

    def __init__(self, config: MoRLLaVAPhiConfig):
        super(MoRLLaVAPhiModel, self).__init__(config)

        # This flag indicates whether the model has already been converted to the MoR structure
        self._is_mor_transformed = False

        # The installed transformers==4.37.2 PhiModel has no model-level `self.rotary_emb`
        # (in the old-style implementation, rotary embedding is attached inside each
        # PhiAttention). But `MoRPhiAttention.forward` (expert_choice_router.py) in the MoR
        # block needs a `position_embeddings=(cos, sin)` (new-style interface) that is
        # precomputed once externally and indexed by position_ids. Therefore we keep a
        # separate model-level rotary embedding module, used only at MoR block positions;
        # ordinary dense layers still use their own internal old-style rotary computation,
        # unaffected by this.
        rotary_dim = int(config.partial_rotary_factor * (config.hidden_size // config.num_attention_heads))
        self.rotary_emb = PhiRotaryEmbedding(
            rotary_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoRBaseModelOutputWithPast]:
        
        # --- Standard input handling (similar to the original PhiModel) ---
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Note: the actual initialization of the KV cache (RecursiveDynamicCache/
        # DynamicCache) is deferred to the "KV Cache initialization (MoR-adapted)" section
        # below, which is the only place that can correctly read num_recursions/
        # sharing_strategy from self.config.mor to compute base_depth. If an empty
        # DynamicCache were initialized here ahead of time, the later
        # `past_key_values is None` check would be skipped, disabling the MoR recursion's
        # KV-sharing logic — so no cache initialization happens here.

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        inputs_embeds = self.embed_dropout(inputs_embeds)

        # Attention mask.
        # if self._use_flash_attention_2:
        #     # 2d mask is passed through the layers
        #     attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        # else:
            # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        
        # --- KV Cache initialization (MoR-adapted) ---
        if use_cache and past_key_values is None:
            if self.config.mor['kv_sharing']['enable']:
                num_recursions = self.config.mor['num_recursions']
                sharing_strategy = self.config.mor['sharing_strategy']

                if sharing_strategy in ["cycle"]:
                    base_depth = self.config.num_hidden_layers // num_recursions
                elif sharing_strategy in ["middle_cycle"]:
                    base_depth = (self.config.num_hidden_layers - 2) // num_recursions
                else:
                    raise ValueError(f"Unknown MoR sharing strategy for KV cache setup: {sharing_strategy}")
                
                past_key_values = RecursiveDynamicCache(
                    base_depth=base_depth, 
                    num_recursion=num_recursions, 
                    sharing=sharing_strategy, 
                    update_cache=self.config.mor['kv_sharing']['update_cache']
                )
            else:
                past_key_values = DynamicCache()
        
        # create position embeddings to be shared across the MoR block positions.
        # The old-style `PhiRotaryEmbedding.forward(x, seq_len)` returns a
        # `[max_seq_len_cached, dim]` cos/sin cache that is not indexed by batch/position;
        # here we manually gather it by `position_ids` into `[bs, seq_len, dim]` to match
        # what MoRPhiAttention (new-style interface) expects.
        rotary_seq_len = int(position_ids.max().item()) + 1 if position_ids.numel() > 0 else seq_length
        cos_cache, sin_cache = self.rotary_emb(hidden_states, seq_len=rotary_seq_len)
        position_embeddings = (cos_cache[position_ids], sin_cache[position_ids])

        # --- MoR core logic ---
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Used to carry state across Expert-choice layers
        prev_selected_tokens = None

        # --- Initialize all diagnostic metrics ---
        # Expert-choice
        total_expert_aux_loss = torch.tensor(0.0, device=hidden_states.device)
        expert_aux_acc_list = []

        # Token-choice (these are usually only computed in one MoR layer)
        total_token_balance_loss = torch.tensor(0.0, device=hidden_states.device)
        token_max_violation = None
        token_router_entropy = None

        # Generic
        total_router_z_loss = torch.tensor(0.0, device=hidden_states.device)
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)

        # Used for entropy regularization: collects the per-token routing
        # activation strength at each MoR block position (depth); after the loop ends
        # these are stacked into [bs, seq_len, num_recursions] for a single unified computation
        depth_activations = []

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Determine whether the current layer is an MoR recursion block
            is_mor_layer = hasattr(decoder_layer, "is_mor_layer") and decoder_layer.is_mor_layer

            if is_mor_layer:
                # Call the forward of the MoR layer
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    prev_selected_tokens=prev_selected_tokens,
                )
                hidden_states = layer_outputs.hidden_state

                # --- Handle different outputs and losses depending on the routing type ---
                if self.config.mor['mor_mode'] == 'expert':
                    prev_selected_tokens = layer_outputs.selected_tokens
                    
                    if hasattr(layer_outputs, "sampling_loss") and layer_outputs.sampling_loss is not None:
                        loss_val = layer_outputs.sampling_loss
                        total_expert_aux_loss += loss_val
                        total_aux_loss += loss_val * self.config.mor['aux_loss_coeff']
                    
                    if hasattr(layer_outputs, "sampling_acc") and layer_outputs.sampling_acc is not None:
                        expert_aux_acc_list.append(layer_outputs.sampling_acc)

                    if hasattr(layer_outputs, "depth_activation") and layer_outputs.depth_activation is not None:
                        depth_activations.append(layer_outputs.depth_activation)
                
                elif self.config.mor['mor_mode'] == 'token':
                    if hasattr(layer_outputs, "balancing_loss") and layer_outputs.balancing_loss is not None:
                        loss_val = layer_outputs.balancing_loss
                        total_token_balance_loss += loss_val
                        total_aux_loss += loss_val * self.config.mor['bal_loss_coeff']

                    if hasattr(layer_outputs, "max_violation") and layer_outputs.max_violation is not None:
                        token_max_violation = layer_outputs.max_violation # usually returned only once

                    if hasattr(layer_outputs, "router_entropy") and layer_outputs.router_entropy is not None:
                        token_router_entropy = layer_outputs.router_entropy # usually returned only once

                # Generic z-loss
                if hasattr(layer_outputs, "router_z_loss") and layer_outputs.router_z_loss is not None:
                    loss_val = layer_outputs.router_z_loss
                    total_router_z_loss += loss_val
                    total_aux_loss += loss_val * self.config.mor['z_loss_coeff']

            else:
                # Call the forward of the standard PhiDecoderLayer. This uses the
                # old-style PhiDecoderLayer from the installed transformers==4.37.2, whose
                # forward signature does not accept `position_embeddings`/`cache_position`
                # (it internally computes rotary using its own per-attention rotary_emb
                # and position_ids, and does not need the model-level precomputed
                # position_embeddings), so these two kwargs must not be passed to it.
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # --- Post-processing and packing the output ---
        # For Expert-choice, compute the average accuracy
        avg_expert_aux_acc = None
        if expert_aux_acc_list:
            avg_expert_aux_acc = torch.stack(expert_aux_acc_list).mean()

        # Assumes the last MoR layer returns dead_token_ratio
        dead_token_ratio = getattr(layer_outputs, 'dead_token_ratio', None) if 'layer_outputs' in locals() and is_mor_layer else None

        # --- Entropy regularization loss ---
        # Stack the per-token routing activation strength at each MoR block position into
        # [bs, seq_len, num_recursions], normalize into a distribution pi(k|x_i), then
        # compute Σ_k pi log(pi) (<=0; closer to 0 means it has degenerated toward "active
        # at only one depth", more negative means activation is more evenly spread out).
        # During training this is added to total_aux_loss scaled by entropy_reg_coeff;
        # minimizing this quantity is equivalent to maximizing entropy and suppressing
        # degenerate collapse.
        total_entropy_loss = torch.tensor(0.0, device=hidden_states.device)
        if depth_activations:
            eps = 1e-8
            stacked_activation = torch.stack(depth_activations, dim=-1)  # [bs, seq_len, num_recursions]
            pi = stacked_activation / (stacked_activation.sum(dim=-1, keepdim=True) + eps)
            entropy_per_token = (pi * torch.log(pi.clamp_min(eps))).sum(dim=-1)
            total_entropy_loss = entropy_per_token.mean()
            total_aux_loss = total_aux_loss + total_entropy_loss * self.config.mor['entropy_reg_coeff']

        hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if not return_dict:
            # When returning a tuple, try to include all information as well
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns, total_aux_loss] if v is not None)

        return MoRBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            # Populate all diagnostic fields
            expert_aux_loss=total_expert_aux_loss,
            expert_aux_acc=avg_expert_aux_acc,
            expert_dead_token_ratio=dead_token_ratio,
            token_balance_loss=total_token_balance_loss,
            token_max_violation=token_max_violation,
            token_router_entropy=token_router_entropy,
            router_z_loss=total_router_z_loss,
            entropy_loss=total_entropy_loss,
            total_aux_loss=total_aux_loss,
        )

# ==========================================================================================
# 3. Implement the MoR version of PhiForCausalLM
#    This is the final model class, responsible for integrating the MoR structure and
#    handling the final loss
# ==========================================================================================

class MoRLLaVAPhiForCausalLM(LlavaMetaForCausalLM, PhiForCausalLM):
    config_class = MoRLLaVAPhiConfig

    def __init__(self, config: MoRLLaVAPhiConfig):
        super(PhiForCausalLM, self).__init__(config)
        self.model = MoRLLaVAPhiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoRCausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True, # force return_dict so aux loss can be accessed
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # Add the auxiliary loss to the main loss
        total_aux_loss = outputs.total_aux_loss
        if loss is not None and total_aux_loss is not None:
            loss += total_aux_loss

        if not return_dict:
            output = (logits,) + (outputs.past_key_values, outputs.hidden_states, outputs.attentions, total_aux_loss)
            return (loss,) + output if loss is not None else output

        # Use the new output class and pass through all detailed metrics
        return MoRCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # --- Pass through all diagnostic metrics ---
            expert_aux_loss=outputs.expert_aux_loss,
            expert_aux_acc=outputs.expert_aux_acc,
            expert_dead_token_ratio=outputs.expert_dead_token_ratio,
            token_balance_loss=outputs.token_balance_loss,
            token_max_violation=outputs.token_max_violation,
            token_router_entropy=outputs.token_router_entropy,
            router_z_loss=outputs.router_z_loss,
            entropy_loss=outputs.entropy_loss,
            total_aux_loss=total_aux_loss,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        # Kept consistent with MoE-LLaVA
        if past_key_values:
            input_ids = input_ids[:, -1:]
        
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images": kwargs.get("images", None),
        })
        return model_inputs

    def initialize_mor_modules(self, model_args):
        """
        This function is called from builder.py to dynamically convert a standard Phi
        model into the MoR structure.
        """
        self.config.mor['mor_enable'] = model_args.mor_enable
        self.config.mor['mor_mode'] = model_args.mor_mode
        self.config.mor['train_modules'] = model_args.train_modules
        self.config.mor['sharing_strategy'] = model_args.sharing_strategy
        self.config.mor['num_recursions'] = model_args.num_recursions
        self.config.mor['group_size'] = getattr(model_args, 'group_size', None)
        self.config.mor['mor_block_init'] = getattr(model_args, 'mor_block_init', 'grouped_sharing')
        self.config.mor['entropy_reg_coeff'] = getattr(model_args, 'entropy_reg_coeff', 0.1)
        self.config.mor['router']['rand_router'] = model_args.rand_router
        self.config.mor['router']['router_type'] = model_args.router_type
        self.config.mor['expert']['cap_warmup_step'] = model_args.cap_warmup_step
        self.config.mor['expert']['expert_capacity'] = model_args.expert_capacity
        self.config.mor['token']['balancing'] = getattr(model_args, 'token_balancing', 'loss')
        self.config.mor['token']['router_func'] = getattr(model_args, 'token_router_func', 'softmax')
        self.config.mor['token']['alpha'] = getattr(model_args, 'token_alpha', 1.0)
        self.config.mor['token']['bal_warmup_step'] = getattr(model_args, 'bal_warmup_step', 0)
        self.config.mor['kv_sharing']['enable'] = model_args.kv_sharing_enable
        self.config.mor['kv_sharing']['update_cache'] = model_args.kv_sharing_update_cache
        self.config.mor['use_aux_loss'] = model_args.use_aux_loss
        self.config.mor['aux_loss_coeff'] = model_args.aux_loss_coeff
        self.config.mor['bal_loss_coeff'] = model_args.bal_loss_coeff
        self.config.mor['use_z_loss'] = model_args.use_z_loss
        self.config.mor['z_loss_coeff'] = model_args.z_loss_coeff
        self.config.mor['gating'] = model_args.gating

        # Skip if already converted
        if getattr(self.model, '_is_mor_transformed', False):
            logger.info("MoR modules already initialized.")
            return

        logger.info("Initializing MoR modules...")

        # If specific training modules are specified, freeze all other parameters
        if self.config.mor['train_modules'] is not None and len(self.config.mor['train_modules']) > 0:
            for n, p in self.named_parameters():
                if any(name in n for name in self.config.mor['train_modules']):
                    continue
                else:
                    p.requires_grad = False
        
        sharing_strategy = self.config.mor['sharing_strategy']
        num_hidden_layers = self.config.num_hidden_layers
        num_recursions = self.config.mor['num_recursions']

        if self.config.mor['mor_mode'] == 'expert':
            capacity = [float(cap) for cap in self.config.mor['expert']['expert_capacity'].split(',')]

            # warmup_step for capacity_factor
            if "cap_warmup_step" in self.config.mor['expert'] and self.config.mor['expert']['cap_warmup_step'] is not None:
                cap_warmup_step = self.config.mor['expert']['cap_warmup_step']
            else:
                cap_warmup_step = self.config.num_warmup_steps * self.config.gradient_accumulation_steps


        # Compute the structure of the recursion block based on the sharing strategy
        mor_block_init = self.config.mor.get('mor_block_init', 'grouped_sharing')

        if sharing_strategy == "cycle":
            if num_hidden_layers % num_recursions != 0:
                raise ValueError("For 'cycle' sharing, num_hidden_layers must be divisible by num_recursions.")
            base_depth = num_hidden_layers // num_recursions

            # [Weight sharing] Construct a single shared submodule M; all num_recursions
            # MoR block positions reuse the same canonical_block object (true parameter-
            # level weight tying, not just a one-time copy at initialization).
            canonical_block = build_shared_recursion_block(
                self.model.layers, num_recursions, base_depth, mor_block_init
            )

            # Instantiate the MoR routing layer
            if self.config.mor['mor_mode'] == 'expert':
                new_layers = nn.ModuleList(
                    [
                        MoRExpertChoiceDecoderLayer(self.config,
                                                    canonical_block,
                                                    self.config.mor, capacity[r_idx] if len(capacity) > 1 else capacity[0], cap_warmup_step,
                        )
                        for r_idx in range(num_recursions)
                    ]
                )
            else: # token
                # For Token-choice, all recursion applications are managed by a single
                # MoR layer that internally iterates the shared canonical_block
                # num_recursions times. Still wrapped in an nn.ModuleList of length 1
                # so that `self.model.layers` remains iterable for the model forward
                # loop, consistent with the expert-choice branch.
                new_layers = nn.ModuleList(
                    [
                        MoRTokenChoiceDecoderLayer(
                            self.config,
                            canonical_block,
                            self.config.mor,
                            bal_warmup_step=self.config.mor['token'].get('bal_warmup_step', 0),
                        )
                    ]
                )

            self.model.layers = new_layers

        elif sharing_strategy == "middle_cycle":
            if (num_hidden_layers - 2) % num_recursions != 0:
                raise ValueError("For 'middle_cycle' sharing, (num_hidden_layers - 2) must be divisible by num_recursions.")
            base_depth = (num_hidden_layers - 2) // num_recursions

            # The recursion block builds the shared submodule M only on the middle layers
            # (first and last layers trimmed off and kept dense)
            middle_layers = self.model.layers[1:-1]
            canonical_block = build_shared_recursion_block(
                middle_layers, num_recursions, base_depth, mor_block_init
            )

            # Instantiate the MoR routing layer
            if self.config.mor['mor_mode'] == 'expert':
                 # For Expert-choice, each recursion step is a separate MoR layer, but they share the same canonical_block
                mor_layers = [
                    MoRExpertChoiceDecoderLayer(
                        self.config,
                        canonical_block,
                        self.config.mor,
                        capacity[r_idx] if len(capacity) > 1 else capacity[0],
                        cap_warmup_step,
                    )
                    for r_idx in range(num_recursions)
                ]
                self.model.layers = nn.ModuleList(
                    [self.model.layers[0]] + mor_layers + [self.model.layers[-1]]
                )
            else: # token
                # For Token-choice, all recursion blocks are managed by a single MoR layer
                mor_layer = MoRTokenChoiceDecoderLayer(
                    self.config,
                    canonical_block,
                    self.config.mor,
                    bal_warmup_step=self.config.mor['token'].get('bal_warmup_step', 0),
                )
                self.model.layers = nn.ModuleList(
                    [self.model.layers[0], mor_layer, self.model.layers[-1]]
                )

        else:
            raise ValueError(f"Unknown MoR sharing strategy: {sharing_strategy}")

        self.model._is_mor_transformed = True
        logger.info(f"Model transformed to MoR structure with type='{self.config.mor['mor_mode']}' and sharing='{sharing_strategy}'.")
        logger.info(f"New model layers: {self.model.layers}")


# ==========================================================================================
# 4. Implement the MoR version of EvalPhiForCausalLM
#    This model class is used for evaluation and is loaded in builder.py
# ==========================================================================================

class EvalMoRLLaVAPhiForCausalLM(PhiForCausalLM):
    config_class = MoRLLaVAPhiConfig

    def __init__(self, config):
        # First, call the parent class's __init__. This creates a standard PhiForCausalLM structure.
        super(EvalMoRLLaVAPhiForCausalLM, self).__init__(config)

        rank0_print("Reconstructing MoR architecture for evaluation...")

        # --- Retrieve all MoR parameters from config ---
        num_hidden_layers = self.config.num_hidden_layers
        num_recursions = self.config.mor['num_recursions']
        sharing_strategy = self.config.mor['sharing_strategy']
        mor_type = self.config.mor['mor_mode']

        if self.config.mor['mor_mode'] == 'expert':
            capacity = [float(cap) for cap in self.config.mor['expert']['expert_capacity'].split(',')]

            # warmup_step for capacity_factor
            if "cap_warmup_step" in self.config.mor['expert'] and self.config.mor['expert']['cap_warmup_step'] is not None:
                cap_warmup_step = self.config.mor['expert']['cap_warmup_step']
            else:
                cap_warmup_step = self.config.num_warmup_steps * self.config.gradient_accumulation_steps


        mor_block_init = self.config.mor.get('mor_block_init', 'grouped_sharing')

        # --- Reconstruct the MoR structure based on config ---
        # This logic must be kept exactly consistent with `initialize_mor_modules`
        # (including the weight-sharing method), otherwise the structure restored from a
        # checkpoint will not match the structure used during training.

        if sharing_strategy == "cycle":
            if num_hidden_layers % num_recursions != 0:
                raise ValueError("For 'cycle' sharing, num_hidden_layers must be divisible by num_recursions.")
            base_depth = num_hidden_layers // num_recursions

            canonical_block = build_shared_recursion_block(
                self.model.layers, num_recursions, base_depth, mor_block_init
            )

            # Instantiate the MoR routing layer
            if self.config.mor['mor_mode'] == 'expert':
                new_layers = nn.ModuleList(
                    [
                        MoRExpertChoiceDecoderLayer(self.config,
                                                    canonical_block,
                                                    self.config.mor, capacity[r_idx] if len(capacity) > 1 else capacity[0], cap_warmup_step,
                        )
                        for r_idx in range(num_recursions)
                    ]
                )
            else: # token
                new_layers = nn.ModuleList(
                    [
                        MoRTokenChoiceDecoderLayer(
                            self.config,
                            canonical_block,
                            self.config.mor,
                            bal_warmup_step=self.config.mor['token'].get('bal_warmup_step', 0),
                        )
                    ]
                )

            self.model.layers = new_layers

        elif sharing_strategy == "middle_cycle":
            if (num_hidden_layers - 2) % num_recursions != 0:
                raise ValueError("For 'middle_cycle' sharing, (num_hidden_layers - 2) must be divisible by num_recursions.")
            base_depth = (num_hidden_layers - 2) // num_recursions

            middle_layers = self.model.layers[1:-1]
            canonical_block = build_shared_recursion_block(
                middle_layers, num_recursions, base_depth, mor_block_init
            )

            # Instantiate the MoR routing layer
            if self.config.mor['mor_mode'] == 'expert':
                 # For Expert-choice, each recursion step is a separate MoR layer, but they share the same canonical_block
                mor_layers = [
                    MoRExpertChoiceDecoderLayer(
                        self.config,
                        canonical_block,
                        self.config.mor,
                        capacity[r_idx] if len(capacity) > 1 else capacity[0],
                        cap_warmup_step,
                    )
                    for r_idx in range(num_recursions)
                ]
                self.model.layers = nn.ModuleList(
                    [self.model.layers[0]] + mor_layers + [self.model.layers[-1]]
                )
            else: # token
                mor_layer = MoRTokenChoiceDecoderLayer(
                    self.config,
                    canonical_block,
                    self.config.mor,
                    bal_warmup_step=self.config.mor['token'].get('bal_warmup_step', 0),
                )
                self.model.layers = nn.ModuleList(
                    [self.model.layers[0], mor_layer, self.model.layers[-1]]
                )

        else:
            raise ValueError(f"Unknown MoR sharing strategy during evaluation init: {sharing_strategy}")
        
        rank0_print(f"Successfully reconstructed MoR structure for evaluation.")
        rank0_print(f"MoR Type: {mor_type}, Sharing Strategy: {sharing_strategy}")
        rank0_print(f"New model layers structure: {self.model.layers}")

# ==========================================================================================
# 4. Register the new model with the Hugging Face AutoClass
# ==========================================================================================
# Note: `EvalMoRLLaVAPhiForCausalLM` is deliberately NOT registered with the
# AutoModelForCausalLM registry. Both classes share the same
# `MoRLLaVAPhiConfig`/model_type="mor_llava_phi"; if both were registered, the later
# `AutoModelForCausalLM.register` call would silently overwrite the earlier one (the HF
# registry keys on the config class, so a given config class can only map to one model
# class), which would cause `AutoModelForCausalLM.from_pretrained`/`from_config` during
# training to actually resolve to `EvalMoRLLaVAPhiForCausalLM` (intended only for
# reconstructing the structure at eval time) instead of the `MoRLLaVAPhiForCausalLM` that
# actually participates in training. `EvalMoRLLaVAPhiForCausalLM` is always constructed
# explicitly via `EvalMoRLLaVAPhiForCausalLM.from_pretrained(...)` in
# moellava/model/builder.py and does not rely on Auto registry discovery, so leaving it
# unregistered is safe and correct.
AutoConfig.register("mor_llava_phi", MoRLLaVAPhiConfig)
AutoModelForCausalLM.register(MoRLLaVAPhiConfig, MoRLLaVAPhiForCausalLM)