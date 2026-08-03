from typing import Callable, List, Optional, Tuple, Union

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from transformers.utils import logging

from .expert_choice_router import MoRPhiAttention, run_mor_block_layer
from .cache_utils import Cache, DynamicCache
from .util import ROUTER_TYPES, MoRLayerOutputWithPast, get_torch_dtype

logger = logging.get_logger(__name__)


class MoRPhiTokenChoiceDecoderLayer(nn.Module):
    """
    Mixture-of-Recurrences Block (Token-choice, a.k.a. Mixture-of-Depths style) for the
    Phi architecture.

    Unlike expert-choice (where each recursion depth is a separate MoRPhiDecoderLayer
    instance and the router decides, at every depth, which tokens are still worth
    computing), token-choice makes a single up-front routing decision per token: each
    token picks exactly one recursion depth in [0, num_recursions) via top-1 routing,
    then is processed sequentially through the shared recursion block that many times.
    Consequently there is only ONE instance of this class per sharing-strategy group
    (it internally iterates over `num_recursions` applications of the same physical
    block), whereas expert-choice instantiates `num_recursions` separate layer objects.
    """

    def __init__(self, config, block: nn.ModuleList, mor_cfg: dict, bal_warmup_step: int = 0):
        super().__init__()
        self.is_mor_layer = True
        self.mor_mode = "token"

        self.config = config
        self.mor_cfg = mor_cfg
        self.bal_warmup_step = bal_warmup_step
        self.training_step = 0

        self.num_recursion = mor_cfg['num_recursions']

        # --- Replace the Attention layer (identical pattern to expert-choice) ---
        # `block` is the single canonical nn.ModuleList shared across all
        # `num_recursions` applications (true parameter-level weight tying). We only
        # need to walk it once here; the `isinstance` guard below makes this safe to
        # call even if expert-choice code already performed the same replacement on
        # this exact object.
        for layer_idx_in_block, original_layer in enumerate(block):
            if isinstance(original_layer.self_attn, MoRPhiAttention):
                continue

            original_attention_module = original_layer.self_attn
            new_attention_module = MoRPhiAttention(
                config=original_attention_module.config,
                layer_idx=original_attention_module.layer_idx,
            )
            new_attention_module.load_state_dict(original_attention_module.state_dict())
            new_attention_module.to(
                device=next(original_attention_module.parameters()).device,
                dtype=next(original_attention_module.parameters()).dtype,
            )
            original_layer.self_attn = new_attention_module
            logger.info(
                f"Replaced PhiAttention with MoRPhiAttention in layer "
                f"{original_layer.self_attn.layer_idx} (block index {layer_idx_in_block})"
            )

        # `self.canonical_block` registers the shared parameters exactly once as a
        # submodule; `self.block_list` is a plain (unregistered) Python list used only
        # for iteration, since it repeats the same already-registered object.
        self.canonical_block = block
        self.block_list = [self.canonical_block for _ in range(self.num_recursion)]

        # --- Router initialization ---
        self.is_random_router = self.mor_cfg['router'].get("rand_router", False)
        if not self.is_random_router:
            router_type = self.mor_cfg['router'].get("router_type", "linear")
            torch_dtype = next(self.canonical_block.parameters()).dtype
            self.router = ROUTER_TYPES[router_type](config, out_dim=self.num_recursion).to(torch_dtype)

        token_cfg = self.mor_cfg.get('token', {})
        self.balancing = token_cfg.get("balancing", "loss")
        if self.balancing == "loss_free":
            self.register_parameter(
                "router_bias", nn.Parameter(torch.zeros(self.num_recursion), requires_grad=False)
            )

    def select_tokens_and_batch_with_padding(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        top_expert_indices: Optional[torch.Tensor] = None,
        index: Optional[int] = None,
        padding_value: float = 0.0,
    ):
        """Gather, for every sample, only the tokens whose chosen depth is still >=
        `index` (i.e. tokens that still need to pass through this depth's block), and
        pad them into a new batch so the shared recursion block can process them
        together in a single call.
        """
        bs, seq_len, hidden_dim = x.shape
        batched_x = []
        selected_batch_indices = []
        selected_seq_indices = []

        for b in range(bs):
            indices = torch.where(top_expert_indices[b] >= index)[0]
            if indices.numel() == 0:
                continue
            selected_batch_indices.append(b)
            selected_seq_indices.append(indices)
            batched_x.append(x[b, indices])

        if len(batched_x) == 0:
            return None, None, None, None, None, None, None

        batched_x = rnn_utils.pad_sequence(
            batched_x, batch_first=True, padding_value=padding_value
        ).to(x.device)
        new_bs, new_seq_len, _ = batched_x.shape

        new_attention_mask = torch.zeros((new_bs, new_seq_len), dtype=x.dtype, device=x.device)
        for b in range(new_bs):
            s = selected_seq_indices[b].numel()
            new_attention_mask[b, :s] = 1

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask_bs = attention_mask.shape[0]
                new_attention_mask = torch.full(
                    (new_bs, 1, new_seq_len, new_seq_len),
                    torch.finfo(attention_mask.dtype).min,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                for b in range(new_bs):
                    orig_b = selected_batch_indices[b] if mask_bs > 1 else 0
                    idx = selected_seq_indices[b]
                    s = idx.numel()
                    sub_mask = attention_mask[orig_b][:, idx][:, :, idx]
                    new_attention_mask[b, :, :s, :s] = sub_mask
            elif attention_mask.dim() == 2:
                pass
            else:
                raise NotImplementedError("Attention mask has unexpected dimensions")

        new_position_ids = None
        if position_ids is not None:
            new_position_ids = torch.arange(
                new_seq_len, dtype=torch.long, device=x.device
            ).unsqueeze(0)

        new_position_embeddings = None
        if position_embeddings is not None:
            cos, sin = position_embeddings
            emb_bs = cos.shape[0]
            head_dim = cos.shape[-1]
            new_cos = torch.zeros((new_bs, new_seq_len, head_dim), dtype=cos.dtype, device=cos.device)
            new_sin = torch.zeros((new_bs, new_seq_len, head_dim), dtype=sin.dtype, device=sin.device)
            for b in range(new_bs):
                src_b = selected_batch_indices[b] if emb_bs > 1 else 0
                idx = selected_seq_indices[b]
                s = idx.numel()
                new_cos[b, :s] = cos[src_b, idx]
                new_sin[b, :s] = sin[src_b, idx]
            new_position_embeddings = (new_cos, new_sin)

        new_cache_position = None
        if cache_position is not None:
            new_cache_position = torch.zeros(
                (new_bs, new_seq_len), dtype=cache_position.dtype, device=cache_position.device
            )
            for b in range(new_bs):
                idx = selected_seq_indices[b]
                s = idx.numel()
                new_cache_position[b, :s] = cache_position[idx]

        return (
            batched_x,
            new_attention_mask,
            new_position_ids,
            new_cache_position,
            new_position_embeddings,
            selected_batch_indices,
            selected_seq_indices,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # Accepted for calling-convention parity with the expert-choice layer, but
        # unused: token-choice is a single MoR layer with no upstream MoR layer to
        # chain a selection from.
        prev_selected_tokens: Optional[torch.LongTensor] = None,
    ) -> MoRLayerOutputWithPast:
        x = hidden_states
        bs, seq_len, hidden_dim = x.shape
        token_cfg = self.mor_cfg.get('token', {})

        if self.training:
            self.training_step += 1

        final_x, updates = x.clone(), x.clone()

        temp = self.mor_cfg.get('temp', 1.0)
        alpha = token_cfg.get("alpha", 1.0)
        router_func = token_cfg.get("router_func", "softmax")

        if not self.is_random_router:
            router_weights = self.router(x / temp)
            if router_func == "sigmoid":
                router_probs = raw_router_probs = torch.sigmoid(router_weights) * alpha
            else:
                router_probs = raw_router_probs = F.softmax(router_weights, dim=-1) * alpha

            if self.balancing == "loss_free":
                router_probs = raw_router_probs + self.router_bias
        else:
            router_weights = torch.rand(bs, seq_len, self.num_recursion, device=x.device, dtype=x.dtype)
            router_probs = raw_router_probs = router_weights * alpha

        if self.training and self.training_step < self.bal_warmup_step:
            # Warm-up: force every token to the deepest recursion so all recursion
            # blocks receive gradient signal before the router starts specializing.
            top_expert_indices = torch.full(
                (bs, seq_len, 1), self.num_recursion - 1, device=x.device, dtype=torch.long
            )
            if self.balancing == "loss_free":
                self.router_bias.data.zero_()
        else:
            _, top_expert_indices = torch.topk(router_probs, 1, dim=-1, sorted=False)

        weights = torch.gather(raw_router_probs, dim=-1, index=top_expert_indices)  # [bs, seq_len, 1]
        top_expert_indices = top_expert_indices.squeeze(-1)  # [bs, seq_len]

        kv_sharing_enabled = self.mor_cfg.get('kv_sharing', {}).get('enable', False)
        gating_mode = self.mor_cfg.get("gating", "weighted")

        outputs = None
        for index, block in enumerate(self.block_list):
            if kv_sharing_enabled:
                # Strategy A: compute the full (unpadded) hidden_states, then gather
                # only the tokens that are still routed to this depth.
                batched_x = x.clone()
                new_attention_mask = attention_mask
                new_position_ids = position_ids
                new_cache_position = cache_position
                new_position_embeddings = position_embeddings
                selected_batch_indices = None
                selected_seq_indices = None
            else:
                (
                    batched_x,
                    new_attention_mask,
                    new_position_ids,
                    new_cache_position,
                    new_position_embeddings,
                    selected_batch_indices,
                    selected_seq_indices,
                ) = self.select_tokens_and_batch_with_padding(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    top_expert_indices=top_expert_indices,
                    index=index,
                )
                if batched_x is None:
                    continue

            kwargs_for_attn = {'selected_tokens': top_expert_indices} if (kv_sharing_enabled and use_cache) else {}

            processed_hidden_states = batched_x
            for layer in block:
                layer_outputs = run_mor_block_layer(
                    layer,
                    processed_hidden_states,
                    attention_mask=new_attention_mask,
                    position_embeddings=new_position_embeddings,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    cache_position=new_cache_position,
                    use_cache=use_cache,
                    **kwargs_for_attn,
                )
                processed_hidden_states = layer_outputs[0]
            outputs = layer_outputs
            batched_x_processed = processed_hidden_states

            if kv_sharing_enabled:
                (
                    batched_x_processed,
                    _,
                    _,
                    _,
                    _,
                    selected_batch_indices,
                    selected_seq_indices,
                ) = self.select_tokens_and_batch_with_padding(
                    batched_x_processed,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    top_expert_indices=top_expert_indices,
                    index=index,
                )
                if selected_batch_indices is None:
                    continue

            for i, batch_idx in enumerate(selected_batch_indices):
                processed_indices = selected_seq_indices[i]
                processed_unpad_x = batched_x_processed[i, : processed_indices.numel()]
                processed_expert_indices = torch.gather(top_expert_indices[batch_idx], dim=0, index=processed_indices)

                finished_indices = torch.where(processed_expert_indices == index)[0]
                finished_indices_in_total = processed_indices[finished_indices]

                finished_x = torch.gather(
                    processed_unpad_x, dim=0, index=finished_indices.view(-1, 1).expand(-1, hidden_dim)
                )
                finished_w = torch.gather(
                    weights[batch_idx], dim=0, index=finished_indices_in_total.view(-1, 1)
                )
                finished_src = finished_x * finished_w if gating_mode == "weighted" else finished_x

                final_x[batch_idx] = torch.scatter_add(
                    final_x[batch_idx],
                    dim=0,
                    index=finished_indices_in_total.view(-1, 1).expand(-1, hidden_dim),
                    src=finished_src.to(x.dtype),
                )

                if index < self.num_recursion - 1:
                    unfinished_indices = torch.where(processed_expert_indices > index)[0]
                    unfinished_indices_in_total = processed_indices[unfinished_indices]

                    unfinished_src = torch.gather(
                        processed_unpad_x, dim=0, index=unfinished_indices.view(-1, 1).expand(-1, hidden_dim)
                    )

                    updates[batch_idx] = torch.scatter(
                        x[batch_idx],
                        dim=0,
                        index=unfinished_indices_in_total.view(-1, 1).expand(-1, hidden_dim),
                        src=unfinished_src.to(x.dtype),
                    )
            x = updates

        balancing_loss = None
        balancing_ratio = None
        router_z_loss = None

        if self.training and not self.is_random_router:
            if self.balancing == "loss":
                P_i = torch.sum(router_probs, dim=(0, 1)) / (bs * seq_len)
                balancing_ratio = torch.bincount(
                    top_expert_indices.reshape(-1), minlength=self.num_recursion
                ) / (bs * seq_len)
                f_i = self.num_recursion * balancing_ratio
                balancing_loss = torch.sum(P_i * f_i)
            elif self.balancing == "loss_free":
                balancing_ratio = torch.bincount(
                    top_expert_indices.reshape(-1), minlength=self.num_recursion
                ) / (bs * seq_len)

            if self.mor_cfg.get("use_z_loss", False):
                router_z_loss = torch.logsumexp(router_weights, dim=-1)
                router_z_loss = torch.square(router_z_loss).mean()

        return MoRLayerOutputWithPast(
            hidden_state=final_x,
            attention_weights=outputs[1:] if outputs is not None else None,
            selected_tokens=None,
            balancing_loss=balancing_loss,
            balancing_ratio=balancing_ratio,
            router_z_loss=router_z_loss,
        )
