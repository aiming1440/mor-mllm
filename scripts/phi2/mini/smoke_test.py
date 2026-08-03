"""
Minimal MoR-MLLM smoke test (pure CPU / PyTorch, no dependency on deepspeed / peft / GPU).

Builds a MoRLLaVAPhiConfig with tiny random weights (very small hidden size / layer count)
and runs the following three checks, for BOTH routing modes ("expert" and "token"):
  1. `MoRLLaVAPhiForCausalLM.initialize_mor_modules` can convert a dense Phi model into the
     MoR structure, and all `num_recursions` MoR block positions **share the exact same
     physical canonical_block object** (true parameter-level weight tying, not just equal
     values at initialization).
  2. A single text-only (no image) forward pass computes LM loss / the mode-specific
     auxiliary losses correctly, and all of them are finite scalars.
  3. `EvalMoRLLaVAPhiForCausalLM` can rebuild, from config alone (without relying on
     model_args), MoR layers with an identical structure (same layer count, per-layer type,
     and shared-submodule sharing relationships).

Usage: python scripts/phi2/mini/smoke_test.py
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mor_mllm.model.language_model.llava_phi_mor import (
    MoRLLaVAPhiConfig,
    MoRLLaVAPhiForCausalLM,
    EvalMoRLLaVAPhiForCausalLM,
)
from mor_mllm.model.mor_modules.expert_choice_router import MoRPhiDecoderLayer
from mor_mllm.model.mor_modules.token_choice_router import MoRPhiTokenChoiceDecoderLayer

MOR_LAYER_CLASSES = {
    "expert": MoRPhiDecoderLayer,
    "token": MoRPhiTokenChoiceDecoderLayer,
}


def build_tiny_config(mor_mode):
    return MoRLLaVAPhiConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        max_position_embeddings=64,
        mor_enable=True,
        mor_mode=mor_mode,
        sharing_strategy="middle_cycle",
        num_recursions=3,
        group_size=2,
        mor_block_init="grouped_sharing",
        routing_threshold=0.8,
        entropy_reg_coeff=0.1,
        router=dict(rand_router=False, router_type="linear"),
        gating="weighted",
        expert=dict(cap_warmup_step=1, expert_capacity="0.5"),
        token=dict(balancing="loss", router_func="softmax", alpha=1.0, bal_warmup_step=0),
        kv_sharing=dict(enable=False, update_cache=False),
        use_aux_loss=True,
        aux_loss_coeff=0.1,
        bal_loss_coeff=0.1,
        use_z_loss=True,
        z_loss_coeff=0.1,
    )


def build_model_args(config, mor_mode):
    return types.SimpleNamespace(
        mor_enable=True,
        mor_mode=mor_mode,
        train_modules=[],
        sharing_strategy="middle_cycle",
        num_recursions=3,
        group_size=2,
        mor_block_init="grouped_sharing",
        entropy_reg_coeff=0.1,
        rand_router=False,
        router_type="linear",
        cap_warmup_step=1,
        expert_capacity="0.5",
        token_balancing="loss",
        token_router_func="softmax",
        token_alpha=1.0,
        bal_warmup_step=0,
        kv_sharing_enable=False,
        kv_sharing_update_cache=False,
        use_aux_loss=True,
        aux_loss_coeff=0.1,
        bal_loss_coeff=0.1,
        use_z_loss=True,
        z_loss_coeff=0.1,
        gating="weighted",
    )


def check_weight_sharing(model, label, mor_mode):
    """Verify that all MoR block positions' canonical_block is the same physical object (`is`, not value equality)."""
    layer_cls = MOR_LAYER_CLASSES[mor_mode]
    mor_layers = [layer for layer in model.model.layers if isinstance(layer, layer_cls)]

    if mor_mode == "expert":
        assert len(mor_layers) == model.config.mor["num_recursions"], (
            f"[{label}] Expected {model.config.mor['num_recursions']} MoR layers, got {len(mor_layers)}"
        )
        block_attr = "block"
    else:
        # Token-choice manages all recursion applications inside a single MoR layer, so
        # there is exactly one such layer regardless of num_recursions.
        assert len(mor_layers) == 1, (
            f"[{label}] Expected exactly 1 token-choice MoR layer, got {len(mor_layers)}"
        )
        block_attr = "canonical_block"

    first_block = getattr(mor_layers[0], block_attr)
    for idx, layer in enumerate(mor_layers[1:], start=1):
        assert getattr(layer, block_attr) is first_block, (
            f"[{label}] MoR block at recursion position {idx} is NOT the same physical "
            f"object as position 0 -- weight sharing is broken."
        )
    # Check that the individual parameter objects are also identical (not just the same ModuleList container)
    if len(mor_layers) > 1:
        for p_name, p in first_block.named_parameters():
            p_other = dict(getattr(mor_layers[1], block_attr).named_parameters())[p_name]
            assert p is p_other, f"[{label}] Parameter '{p_name}' is not shared by identity."

    print(f"[{label}] OK: {len(mor_layers)} MoR layer(s) share the same physical "
          f"canonical_block ({sum(p.numel() for p in first_block.parameters())} params).")


def check_first_last_dense(model, label, mor_mode):
    layer_cls = MOR_LAYER_CLASSES[mor_mode]
    layers = model.model.layers
    assert not isinstance(layers[0], layer_cls), f"[{label}] First layer must stay dense."
    assert not isinstance(layers[-1], layer_cls), f"[{label}] Last layer must stay dense."
    print(f"[{label}] OK: first/last layers kept dense (not converted to MoR).")


def run_for_mode(mor_mode):
    print(f"\n=== Running smoke test for mor_mode='{mor_mode}' ===")
    torch.manual_seed(0)

    # --- 1. Build a dense model and convert it to the MoR structure ---
    config = build_tiny_config(mor_mode)
    model = MoRLLaVAPhiForCausalLM(config)
    model.initialize_mor_modules(model_args=build_model_args(config, mor_mode))
    model.eval()

    check_first_last_dense(model, "train-side model", mor_mode)
    check_weight_sharing(model, "train-side model", mor_mode)

    # --- 2. Forward pass: check that LM loss / mode-specific aux losses are all computed correctly ---
    batch_size, seq_len = 2, 12
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Token-choice routing/backward needs gradients to flow, so run this in train mode.
    model.train()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        images=None,
        use_cache=False,
        return_dict=True,
    )

    common_fields = ["loss", "router_z_loss", "entropy_loss", "total_aux_loss"]
    mode_fields = ["expert_aux_loss"] if mor_mode == "expert" else ["token_balance_loss"]
    for field_name in ["loss", "total_aux_loss"] + mode_fields:
        value = getattr(outputs, field_name)
        assert value is not None, f"Output field '{field_name}' is None."
        assert value.dim() == 0, f"Output field '{field_name}' is not a scalar: shape={value.shape}"
        assert torch.isfinite(value).all(), f"Output field '{field_name}' is not finite: {value}"

    aux_field = "expert_aux_loss" if mor_mode == "expert" else "token_balance_loss"
    print(f"Forward OK: loss={outputs.loss.item():.4f}, "
          f"{aux_field}={getattr(outputs, aux_field).item():.4f}, "
          f"total_aux_loss={outputs.total_aux_loss.item():.4f}")

    # Also run backward, to confirm gradients flow correctly through the shared submodule and the router
    outputs.loss.backward()
    layer_cls = MOR_LAYER_CLASSES[mor_mode]
    mor_layer = next(layer for layer in model.model.layers if isinstance(layer, layer_cls))
    block = mor_layer.block if mor_mode == "expert" else mor_layer.canonical_block
    router_grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all()
                          for p in mor_layer.router.parameters())
    block_grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all()
                         for p in block.parameters())
    assert router_grad_ok, "Router received no finite gradient."
    assert block_grad_ok, "Shared block received no finite gradient."
    print("Backward OK: router and shared block both received finite gradients.")
    model.eval()

    # --- 3. Eval-side reconstruction: rebuild structurally identical MoR layers from config alone ---
    eval_model = EvalMoRLLaVAPhiForCausalLM(config)
    eval_model.eval()

    check_first_last_dense(eval_model, "eval-side reconstructed model", mor_mode)
    check_weight_sharing(eval_model, "eval-side reconstructed model", mor_mode)

    train_layer_types = [type(layer).__name__ for layer in model.model.layers]
    eval_layer_types = [type(layer).__name__ for layer in eval_model.model.layers]
    assert train_layer_types == eval_layer_types, (
        f"Layer structure mismatch: train={train_layer_types} vs eval={eval_layer_types}"
    )
    print(f"Reconstruction OK: eval-side layer structure matches train-side: {eval_layer_types}")


def main():
    for mor_mode in ["expert", "token"]:
        run_for_mode(mor_mode)

    print("\nAll smoke test checks passed.")


if __name__ == "__main__":
    main()
