from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import LlamaConfig
from transformers.utils.deprecation import deprecate_kwarg

from model.kv_caches.cache_utils import Cache, StaticCache, DynamicCache, RecursiveDynamicCache
from model.base_model.modeling_llama import (
    LlamaModel, 
    LlamaForCausalLM, 
    KwargsForCausalLM,
    LlamaRMSNorm,
    LLAMA_START_DOCSTRING,
    LLAMA_INPUTS_DOCSTRING,
)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


@dataclass
class MoRBaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    sampling_loss: Optional[torch.FloatTensor] = None
    sampling_acc: Optional[torch.FloatTensor] = None
    sampling_topk_acc: Optional[torch.FloatTensor] = None
    uniformity: Optional[torch.FloatTensor] = None
    dead_token_seq: Optional[torch.FloatTensor] = None
    balancing_loss: Optional[torch.FloatTensor] = None
    balancing_ratio: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None


@dataclass
class MoRCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    sampling_loss: Optional[torch.FloatTensor] = None
    sampling_acc: Optional[torch.FloatTensor] = None
    sampling_topk_acc: Optional[torch.FloatTensor] = None
    uniformity: Optional[torch.FloatTensor] = None
    dead_token_seq: Optional[torch.FloatTensor] = None
    balancing_loss: Optional[torch.FloatTensor] = None
    balancing_ratio: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None
    
    
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class MoRLlamaModel(LlamaModel):
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, MoRBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            if "kv_sharing" in self.config and self.config.kv_sharing is not None:
                kwargs = self.config.kv_sharing
                past_key_values = RecursiveDynamicCache(kwargs["base_depth"], kwargs["num_recursion"], kwargs["sharing"], kwargs["update_cache"])
            else:
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        prev_selected_tokens = None
        sampling_loss = torch.tensor(0.0, device=hidden_states.device)
        sampling_acc_list = []
        sampling_topk_acc_list = []
        uniformity = None # torch.tensor(0.0, device=hidden_states.device)
        dead_token_seq = None # torch.tensor([0.0] * self.config.max_position_embeddings, device=hidden_states.device)
        balancing_loss = torch.tensor(0.0, device=hidden_states.device)
        balancing_ratio = torch.tensor(0.0, device=hidden_states.device)
        router_z_loss = torch.tensor(0.0, device=hidden_states.device)
        
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # TODO: support MoRLlamaDecoderLayer
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                if hasattr(decoder_layer, "mor") and decoder_layer.mor:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        prev_selected_tokens=prev_selected_tokens,
                        **flash_attn_kwargs,
                    )
                    if decoder_layer.mor_type == "expert":
                        prev_selected_tokens = layer_outputs.selected_tokens
                        if layer_outputs.sampling_loss is not None:
                            sampling_loss += layer_outputs.sampling_loss
                        if layer_outputs.sampling_acc is not None:
                            sampling_acc_list.append(layer_outputs.sampling_acc)
                        if layer_outputs.sampling_topk_acc is not None:
                            sampling_topk_acc_list.append(layer_outputs.sampling_topk_acc)
                        # if layer_outputs.uniformity is not None:
                        #     uniformity += layer_outputs.uniformity
                        # if layer_outputs.dead_token_seq is not None:
                        #     dead_token_seq = layer_outputs.dead_token_seq
                        if layer_outputs.router_z_loss is not None:
                            router_z_loss += layer_outputs.router_z_loss
                            
                    elif decoder_layer.mor_type == "token":
                        if layer_outputs.balancing_loss is not None:
                            balancing_loss = layer_outputs.balancing_loss
                        if layer_outputs.balancing_ratio is not None:
                            balancing_ratio = layer_outputs.balancing_ratio
                        if layer_outputs.router_z_loss is not None:
                            router_z_loss = layer_outputs.router_z_loss
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = MoRBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            sampling_loss=sampling_loss,
            sampling_acc=sum(sampling_acc_list)/len(sampling_acc_list) if len(sampling_acc_list) > 0 else torch.tensor(0.0, device=hidden_states.device),
            sampling_topk_acc=sum(sampling_topk_acc_list)/len(sampling_topk_acc_list) if len(sampling_topk_acc_list) > 0 else torch.tensor(0.0, device=hidden_states.device),
            uniformity=uniformity,
            dead_token_seq=dead_token_seq,
            balancing_loss=balancing_loss,
            balancing_ratio=balancing_ratio,
            router_z_loss=router_z_loss,
        )
        return output if return_dict else output.to_tuple()


class MoRLlamaForCausalLM(LlamaForCausalLM):
    
    def __init__(self, config):
        super().__init__(config)
        self.model = MoRLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def transform_layer_to_mor_expert(self, cfg):
        from model.mor_model.expert_choice_router import MoRLlamaDecoderLayer
        
        capacity = [float(cap) for cap in cfg.mor.capacity.split(',')]
        # warmup_step for capacity_factor
        if "cap_warmup_step" in cfg.mor.expert and cfg.mor.expert.cap_warmup_step is not None:
            cap_warmup_step = cfg.mor.expert.cap_warmup_step
        else:
            cap_warmup_step = cfg.num_warmup_steps * cfg.gradient_accumulation_steps
        
        sharing = cfg.recursive.sharing
        num_recursion = cfg.recursive.num_recursion        
        num_hidden_layers = len(self.model.layers)
        
        # Cycle sharing is for early-exiting mechanism
        if sharing == "cycle":
            base_depth = num_hidden_layers // num_recursion
            self.model.layers = nn.ModuleList(
                [
                    MoRLlamaDecoderLayer(self.config, nn.ModuleList([self.model.layers[layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]), 
                                         cfg, capacity[recur_idx], cap_warmup_step,) 
                    for recur_idx in range(num_recursion)
                ]
            )
        elif sharing == "middle_cycle":
            base_depth = (num_hidden_layers - 2) // num_recursion
            self.model.layers = nn.ModuleList(
                [self.model.layers[0]] + \
                [
                    MoRLlamaDecoderLayer(self.config, nn.ModuleList([self.model.layers[1 + layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]), 
                                         cfg, capacity[recur_idx], cap_warmup_step,)
                    for recur_idx in range(num_recursion)
                ]
                + [self.model.layers[-1]]
            )
    
    def transform_layer_to_mor_token(self, cfg):
        from model.mor_model.token_choice_router import MoRLlamaDecoderLayer
                
        # warmup_step for balancing
        bal_warmup_step = 0
        if "bal_warmup_step" in cfg.mor.token and cfg.mor.token.bal_warmup_step > 0:
            bal_warmup_step = cfg.mor.token.bal_warmup_step * cfg.gradient_accumulation_steps
        
        sharing = cfg.recursive.sharing
        num_recursion = cfg.recursive.num_recursion        
        num_hidden_layers = len(self.model.layers)
        
        # Cycle sharing is for early-exiting mechanism
        if sharing == "cycle":
            base_depth = num_hidden_layers // num_recursion
            self.model.layers = MoRLlamaDecoderLayer(
                self.config,
                nn.ModuleList([nn.ModuleList([self.model.layers[layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]) for recur_idx in range(num_recursion)]),
                cfg,
                bal_warmup_step,
            )
        elif sharing == "middle_cycle":
            base_depth = (num_hidden_layers - 2) // num_recursion
            self.model.layers = nn.ModuleList(
                [self.model.layers[0]] + \
                [MoRLlamaDecoderLayer(
                    self.config,
                    nn.ModuleList([nn.ModuleList([self.model.layers[1 + layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]) for recur_idx in range(num_recursion)]),
                    cfg,
                    bal_warmup_step,
                ),] + \
                [self.model.layers[-1]]
            )
    
    def set_kv_sharing_config(self, cfg):
        if cfg.kv_sharing.sharing in ["cycle", "sequence"]:
            base_depth = self.config.num_hidden_layers // cfg.kv_sharing.num_recursion
        elif cfg.kv_sharing.sharing in ["middle_cycle"]:
            base_depth = (self.config.num_hidden_layers - 2) // cfg.kv_sharing.num_recursion
        
        if "kv_sharing" in cfg:                
            kwargs = {
                "enable": cfg.kv_sharing.enable,
                "base_depth": base_depth,
                "num_recursion": cfg.kv_sharing.num_recursion,
                "sharing": cfg.kv_sharing.sharing,
                "update_cache": cfg.kv_sharing.update_cache if "update_cache" in cfg.kv_sharing else False,
            }        
            self.model.config.kv_sharing = kwargs
        else:
            self.model.config.kv_sharing = None
                    
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoRCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, MoRCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoRCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sampling_loss=outputs.sampling_loss,
            sampling_acc=outputs.sampling_acc,
            sampling_topk_acc=outputs.sampling_topk_acc,
            uniformity=outputs.uniformity,
            dead_token_seq=outputs.dead_token_seq,
            balancing_loss=outputs.balancing_loss,
            balancing_ratio=outputs.balancing_ratio,
            router_z_loss=outputs.router_z_loss,
        )
