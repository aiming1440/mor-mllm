from typing import Callable, List, Optional, Tuple, Union

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import logging
try:
    # `ALL_ATTENTION_FUNCTIONS` only exists in newer transformers versions (after the
    # attention-interface refactor). The pinned environment uses transformers==4.37.2,
    # which lacks this symbol. Since it's only actually used below when
    # `config._attn_implementation != "eager"` (the default is "eager", which goes
    # through `eager_attention_forward`), we allow it to be missing here and only
    # raise when that branch is actually hit.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None
from transformers.models.phi.modeling_phi import PhiAttention, apply_rotary_pos_emb

from ..mor_modules.cache_utils import Cache, DynamicCache
from ..mor_modules.util import ROUTER_TYPES, MoRLayerOutputWithPast
try:
    # `modeling_llama.py` is a reference implementation ported wholesale from a newer
    # transformers version; many of its module-level imports (StaticCache/
    # FlashAttentionKwargs/ROPE_INIT_FUNCTIONS/...) don't exist under the pinned
    # transformers==4.37.2. We only need `eager_attention_forward` from that file
    # (a pure-torch implementation with no dependency on those newer symbols), so we
    # use try/except as a fallback: if the import fails, fall back to the equivalent
    # implementation defined locally below, so that unrelated import errors in that
    # reference file don't break loading of the whole MoR module.
    from ..base_model.modeling_llama import eager_attention_forward
except ImportError:
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        num_key_value_groups = getattr(module, "num_key_value_groups", 1)
        key_states = repeat_kv(key, num_key_value_groups)
        value_states = repeat_kv(value, num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights
from .util import get_torch_dtype


from .cache_utils import Cache

logger = logging.get_logger(__name__)


def run_mor_block_layer(
    layer,
    hidden_states,
    attention_mask,
    position_embeddings,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    **extra_attn_kwargs,
):
    """Manually runs layernorm -> self_attn -> mlp -> residual connection, following the
    new calling convention that applies after `MoRPhiDecoderLayer.__init__` replaces
    self_attn.

    Background: the elements of `self.block` are the old-style `PhiDecoderLayer` from
    the local `phi/modeling_phi.py`, whose own `forward` only accepts `position_ids`
    (computing cos/sin internally) and assumes `self_attn` returns a 3-tuple
    `(attn_output, attn_weights, present_key_value)`. However,
    `MoRPhiDecoderLayer.__init__` has already replaced `layer.self_attn` with the
    new-style `MoRPhiAttention` (which requires the caller to precompute
    `position_embeddings=(cos, sin)` already gathered at the top_k token positions,
    and returns only a 2-tuple `(attn_output, attn_weights)`). Calling `layer(...)`
    directly (which goes through the old-style `PhiDecoderLayer.forward`) can neither
    pass in `position_embeddings` nor match the new-style self_attn's return arity.
    So here we bypass the old-style `PhiDecoderLayer.forward` and directly reuse its
    `input_layernorm`/`mlp`/`resid_dropout` submodules, manually calling `self_attn`
    according to the new-style convention.
    """
    residual = hidden_states
    normed_hidden_states = layer.input_layernorm(hidden_states)

    attn_output, attn_weights = layer.self_attn(
        hidden_states=normed_hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_value=past_key_value,
        cache_position=cache_position,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **extra_attn_kwargs,
    )
    attn_output = layer.resid_dropout(attn_output)

    feed_forward_hidden_states = layer.resid_dropout(layer.mlp(normed_hidden_states))
    hidden_states = attn_output + feed_forward_hidden_states + residual

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)
    if use_cache:
        outputs += (past_key_value,)
    return outputs


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """New-style rotary application function (position_ids already indexed externally),
    overriding the old-style `apply_rotary_pos_emb` imported above from
    transformers.models.phi.modeling_phi (old-style signature is
    `(q, k, cos, sin, position_ids, unsqueeze_dim=1)`, and requires `cos`/`sin` to be
    the un-batch-indexed `[max_seq_len, dim]` cache).

    In this file, `MoRPhiAttention.forward` receives `position_embeddings=(cos, sin)`
    as `[bs, seq_len, dim]` tensors already gathered by `position_ids` in
    `MoRLLaVAPhiModel.forward` (consistent with the rotary interface of newer
    transformers Llama/Phi3 versions), so we use the new-style implementation here,
    which no longer needs an extra position_ids argument.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MoRPhiAttention(PhiAttention):
    """
    A wrapper for PhiAttention that supports sparse KV cache updates for the
    'Recursive KV Sharing' strategy in Mixture-of-Recurrences.

    This class inherits from the original PhiAttention and overrides the forward
    method. The core change is that when past_key_value.update is called, if kwargs
    contains 'selected_tokens', it triggers the sparse update logic inside the Cache
    object.
    """

    def __init__(self, config, layer_idx: int):
        # Call the parent constructor directly, inheriting all weights and attributes
        super().__init__(config, layer_idx)
        # The installed transformers==4.37.2 version of PhiAttention only records the
        # partial-rotary dimension on `self.rotary_emb.dim`, without a separate
        # `self.rotary_ndims` attribute (that naming only exists in newer transformers
        # versions); we add it here for use in forward below.
        self.rotary_ndims = self.rotary_emb.dim
        # Similarly, the old PhiAttention computes the attention scale inline inside
        # forward as `1 / math.sqrt(head_dim)`, without storing it as a `self.scaling`
        # attribute (the attention-interface refactor in newer transformers versions
        # requires the module to have this attribute for
        # `eager_attention_forward`/`ALL_ATTENTION_FUNCTIONS`); we add it here.
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        cos, sin = position_embeddings
        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_ndims],
            query_states[..., self.rotary_ndims :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_ndims],
            key_states[..., self.rotary_ndims :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            # This class is for hybrid KV sharing that leverages shared caches 
            # for inactive positions while updating active ones through actual computation
            if "selected_tokens" in kwargs:
                cache_kwargs["selected_tokens"] = kwargs["selected_tokens"]

            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            elif ALL_ATTENTION_FUNCTIONS is not None:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            else:
                raise ImportError(
                    f"config._attn_implementation='{self.config._attn_implementation}' requires "
                    "transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS, which is not available in "
                    "the installed transformers version. Use attn_implementation='eager' instead, or "
                    "upgrade transformers."
                )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)
        return attn_output, attn_weights   


class MoRPhiDecoderLayer(nn.Module):
    """
    Mixture-of-Recurrences Block (Expert-choice) for Phi Architecture.
    This module wraps a group of standard PhiDecoderLayers, and uses a router to
    decide which tokens are computed within this recursion block.
    """
    def __init__(self, config, block: nn.ModuleList, mor_cfg: dict, capacity_factor=1.0, cap_warmup_step=0,):
        super().__init__()
        self.is_mor_layer = True
        self.mor_mode = "expert"
        
        self.config = config
        self.block = block  # This is an nn.ModuleList containing one or more PhiDecoderLayers
        self.mor_cfg = mor_cfg
        self.capacity_factor = capacity_factor
        self.cap_warmup_step = cap_warmup_step  # Number of warm-up steps for capacity_factor

        # --- Replace the Attention layer ---
        # Iterate over all original PhiDecoderLayers in the recursion block
        for layer_idx_in_block, original_layer in enumerate(self.block):
            # Check whether self_attn is already MoRPhiAttention, to avoid replacing it twice
            if isinstance(original_layer.self_attn, MoRPhiAttention):
                continue

            # 1. Save the original attention module
            original_attention_module = original_layer.self_attn

            # 2. Create a new MoRPhiAttention instance.
            #    Its structure will be identical to the original module, since they use
            #    the same config and layer_idx.
            new_attention_module = MoRPhiAttention(
                config=original_attention_module.config,
                layer_idx=original_attention_module.layer_idx
            )

            # 3. [Core step] Load the weights!
            #    Load the original module's state dict (containing all pretrained weights)
            #    into the new module.
            new_attention_module.load_state_dict(original_attention_module.state_dict())

            # 4. Move the new module to the same device and dtype as the original module, just in case
            new_attention_module.to(
                device=next(original_attention_module.parameters()).device,
                dtype=next(original_attention_module.parameters()).dtype
            )

            # 5. Replace the original module with the new module that has the loaded weights.
            original_layer.self_attn = new_attention_module

            # Log to confirm the replacement succeeded
            logger.info(f"Replaced PhiAttention with MoRPhiAttention in layer {original_layer.self_attn.layer_idx} (block index {layer_idx_in_block})")

        logger.info(f"Replaced PhiAttention with MoRPhiAttention for layers in the block.")


        # --- Router initialization ---
        # Allow random routing for debugging
        self.is_random_router = self.mor_cfg['router'].get("rand_router", False)
        if not self.is_random_router:
            router_type = self.mor_cfg['router'].get("router_type", "linear")
            self.router = ROUTER_TYPES[router_type](config)

        # --- Auxiliary loss related ---
        # Expert-choice uses BCEWithLogitsLoss to compute the auxiliary routing loss
        self.use_aux_loss = self.mor_cfg.get("use_aux_loss", True)
        if self.use_aux_loss:
            self.aux_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        self.training_step = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # MoR-specific argument
        prev_selected_tokens: Optional[torch.LongTensor] = None,
    ) -> MoRLayerOutputWithPast:

        batch_size, total_seq_len, hidden_dim = hidden_states.shape

        # --- Step 0: Prepare inputs ---
        # If there's a routing decision from the previous MoR layer, first filter the
        # current layer's input based on it. This is key to implementing multi-level
        # routing.
        if prev_selected_tokens is not None:
            # `prev_selected_tokens` has shape [batch_size, prev_k, 1]
            # We use it to gather the corresponding tokens from `hidden_states`
            current_hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=prev_selected_tokens.expand(-1, -1, hidden_dim)
            )
        else:
            current_hidden_states = hidden_states

        current_seq_len = current_hidden_states.shape[1]

        # --- Step 1: Routing decision ---
        # Dynamically adjust the capacity factor based on the current training step
        if self.training:
            self.training_step += 1
            if self.cap_warmup_step > 0:
                step_ratio = min(1.0, self.training_step / self.cap_warmup_step)
                # Use cosine decay to smoothly transition from 1.0 to capacity_factor
                decay_factor = 0.5 * (1.0 + math.cos(math.pi * step_ratio))
                capacity_factor = self.capacity_factor + (1.0 - self.capacity_factor) * decay_factor
            else:
                capacity_factor = self.capacity_factor
        else:
            capacity_factor = self.capacity_factor

        # Compute the number of top-k tokens to select in this layer
        top_k = max(1, int(capacity_factor * current_seq_len))

        if self.is_random_router:
            # Random routing, used for testing and as a baseline comparison
            router_logits = torch.rand(batch_size, current_seq_len, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            # Use the learned router to compute a score for each token
            router_logits = self.router(current_hidden_states) # Shape: [bs, current_seq_len, 1]

        # Select the top-k tokens by score
        # `selected_scores` and `indices_in_current` both have shape [bs, top_k, 1]
        selected_scores, indices_in_current = torch.topk(router_logits, top_k, dim=1, sorted=False)

        # [Key point] Sort the indices to preserve the causal order required by attention
        indices_in_current, sort_order = torch.sort(indices_in_current, dim=1)
        selected_scores = torch.gather(selected_scores, dim=1, index=sort_order)

        # --- Step 2: Compute the auxiliary loss ---
        sampling_loss = None
        sampling_acc = None
        sampling_topk_acc = None
        router_z_loss = None

        if self.training:
            if self.use_aux_loss and not self.is_random_router:
                # Create a one-hot target tensor
                targets = torch.zeros_like(router_logits, dtype=router_logits.dtype)
                targets.scatter_(1, indices_in_current, 1.0)

                # Compute the auxiliary loss, encouraging the router to predict which
                # tokens end up selected
                sampling_loss = self.aux_loss_fn(router_logits, targets)

                # Convert logits to probabilities and threshold at 0.5 for predictions
                predictions = (torch.sigmoid(router_logits) >= 0.5)
                # Compute how well predictions match the targets
                correct_predictions = (predictions == targets.bool())
                # Compute the average accuracy
                sampling_acc = correct_predictions.float().mean()

                sampling_topk_acc = None

            if self.mor_cfg.get("use_z_loss", False):
                # z_loss penalizes large absolute logit values, helping stabilize training
                router_z_loss = torch.square(router_logits).mean()

        # --- Step 3: Prepare inputs for computation ---
        # `indices_in_current` are relative indices into `current_hidden_states`
        # (which may have already been reduced). We need to convert them into absolute
        # indices into the original `hidden_states`.
        if prev_selected_tokens is not None:
            # Chained indexing: gather the current layer's indices again from the
            # previous layer's absolute indices
            absolute_indices = torch.gather(prev_selected_tokens, dim=1, index=indices_in_current)
        else:
            absolute_indices = indices_in_current

        # --- Step 4: Forward through the block (core computation) ---
        # Execute different computation paths depending on the chosen KV cache strategy

        kv_sharing_enabled = self.mor_cfg.get('kv_sharing', {}).get('enable', False)

        if kv_sharing_enabled:
            # === Strategy A: Recursive KV Sharing ===
            # Under this strategy, we compute all tokens, but only update the
            # hidden_states of the selected tokens. This avoids having to handle a
            # sparse KV cache, making the implementation simpler and more efficient.

            # Compute using the full hidden_states
            block_input = hidden_states

            # Pass through the full attention_mask and position_ids
            current_attention_mask = attention_mask
            current_position_ids = position_ids

            # Pass the absolute indices to the attention layer for use when updating
            # the KV cache
            # (this requires a small modification to the PhiAttention layer to accept
            # this argument)
            kwargs_for_attn = {'selected_tokens': absolute_indices} if use_cache else {}
            
            processed_hidden_states = block_input

            for layer in self.block:
                layer_outputs = run_mor_block_layer(
                    layer,
                    processed_hidden_states,
                    attention_mask=current_attention_mask,
                    position_embeddings=position_embeddings,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                    use_cache=use_cache,
                    **kwargs_for_attn,
                )
                processed_hidden_states = layer_outputs[0]

            # From the full output, gather only the results for the selected tokens,
            # for use in the subsequent scatter_add
            top_k_processed_states = torch.gather(
                processed_hidden_states,
                dim=1,
                index=absolute_indices.expand(-1, -1, hidden_dim)
            )

        else:
            # === Strategy B: Recursion-wise KV Caching ===
            # This is a more precise but more complex-to-implement strategy. All
            # inputs must be indexed.

            # 1. Index hidden_states
            top_k_hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=absolute_indices.expand(-1, -1, hidden_dim)
            )

            # 2. [Key point] Index attention_mask
            # LLaVA's mask is 4D: [bs, 1, seq_len, seq_len]
            # We need to index both the 2nd dim (query) and 3rd dim (key/value)
            if attention_mask is not None and attention_mask.dim() == 4:
                # Index the query dimension (dim=2)
                mask_q_indexed = torch.gather(
                    attention_mask, 2,
                    absolute_indices.unsqueeze(1).expand(-1, 1, -1, total_seq_len)
                )
                # Index the key/value dimension (dim=3): for each selected query row,
                # also keep only the selected top_k positions along the key dimension.
                # `absolute_indices` is [bs, top_k, 1] -> squeezed to [bs, top_k], then
                # broadcast across all top_k query rows (equivalent to all rows sharing
                # the same set of key indices), producing a gather index of shape
                # [bs, 1, top_k, top_k].
                key_indices = absolute_indices.squeeze(-1)  # [bs, top_k]
                current_attention_mask = torch.gather(
                    mask_q_indexed, 3,
                    key_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, top_k, -1)
                )
            else:
                current_attention_mask = attention_mask # If mask is None or 2D, leave it unprocessed for now

            # 3. [Key point] Index position_ids
            # position_ids is theoretically [bs, seq_len], but upstream
            # (LlavaMetaModel/PhiModel) commonly uses a batch-independent `[1, seq_len]`
            # (all samples share the same set of position indices). We first broadcast
            # it to the actual batch_size before gathering, to avoid a batch-dimension
            # mismatch error.
            if position_ids is not None:
                position_ids_expanded = position_ids.expand(batch_size, -1) if position_ids.shape[0] == 1 else position_ids
                current_position_ids = torch.gather(position_ids_expanded, 1, absolute_indices.squeeze(-1))
            else:
                current_position_ids = None

            # 3b. [Key point] Index position_embeddings (cos, sin) in sync: hidden_states
            # has already been reduced to only top_k tokens, so the rotary cos/sin must
            # also be reduced to the same top_k positions, otherwise they won't match
            # the seq_len dimension of query/key.
            cos, sin = position_embeddings
            cos = cos.expand(batch_size, -1, -1) if cos.shape[0] == 1 else cos
            sin = sin.expand(batch_size, -1, -1) if sin.shape[0] == 1 else sin
            rot_index = absolute_indices.expand(-1, -1, cos.shape[-1])
            current_position_embeddings = (
                torch.gather(cos, 1, rot_index),
                torch.gather(sin, 1, rot_index),
            )

            # 4. Call the internal block to perform the computation
            processed_hidden_states = top_k_hidden_states

            for layer in self.block:
                layer_outputs = run_mor_block_layer(
                    layer,
                    processed_hidden_states,
                    attention_mask=current_attention_mask,
                    position_embeddings=current_position_embeddings,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                processed_hidden_states = layer_outputs[0]
            
            top_k_processed_states = processed_hidden_states

        # --- Step 5: Merge results (scatter-add) ---
        # Use scatter_add to write the computed tokens back into the original
        # hidden_states tensor
        # First, create a zero tensor as the base for scatter_add
        output_hidden_states = torch.zeros_like(hidden_states)

        # Optionally weight the output by the router score
        if self.mor_cfg.get("gating", "weighted") == "weighted":
            # Use sigmoid to convert logits into weights between 0 and 1
            gating_weights = torch.sigmoid(selected_scores)
            scatter_src = top_k_processed_states * gating_weights
        else:
            scatter_src = top_k_processed_states

        output_hidden_states.scatter_add_(
            dim=1,
            index=absolute_indices.expand(-1, -1, hidden_dim),
            src=scatter_src
        )

        # Add the residual connection (i.e. unselected tokens are kept unchanged)
        # Create a mask identifying which tokens were not selected
        not_selected_mask = torch.ones_like(hidden_states, dtype=torch.bool)
        not_selected_mask.scatter_(1, absolute_indices.expand(-1, -1, hidden_dim), False)

        # Copy unselected tokens directly from the original input
        output_hidden_states = torch.where(not_selected_mask, hidden_states, output_hidden_states)

        # --- Step 6: Record this layer's routing activation strength (used for
        # model-level entropy regularization, Eq. 11-12) ---
        # Unselected tokens have an activation strength of 0 at this depth position;
        # selected tokens record their sigmoid(router_score) as this token's
        # "participation strength" at this depth. The depth_activation from multiple
        # MoR blocks is stacked in `MoRLLaVAPhiModel.forward` into
        # [bs, seq_len, num_recursions], and after normalization is used to compute
        # Σ_k π(k|x_i) log π(k|x_i).
        depth_activation = torch.zeros(
            batch_size, total_seq_len, device=hidden_states.device, dtype=hidden_states.dtype
        )
        depth_activation.scatter_(
            1, absolute_indices.squeeze(-1), torch.sigmoid(selected_scores).squeeze(-1)
        )

        return MoRLayerOutputWithPast(
            hidden_state=output_hidden_states,
            attention_weights=layer_outputs[1:],
            selected_tokens=absolute_indices,
            sampling_loss=sampling_loss,
            sampling_acc=sampling_acc,
            sampling_topk_acc=sampling_topk_acc,
            router_z_loss=router_z_loss,
            depth_activation=depth_activation,
        )

