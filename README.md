# MoR-MLLM

MoR-MLLM applies **Mixture-of-Recursions (MoR)** — sharing a single group of
transformer layers across multiple recursion "depths" and routing tokens
dynamically among those depths — to a multimodal (vision + language) LLaVA-style
model.

- **Expert-choice routing**: at each recursion depth, a router selects the
  top-k tokens (by capacity factor) worth computing further.
- **Token-choice routing** (Mixture-of-Depths style): each token picks, once,
  the total recursion depth it will be processed to, via top-1 routing.

## Project layout

```
mor_mllm/
  model/
    language_model/
      llava_phi_mor.py       # MoRLLaVAPhiConfig
      llava_phi.py           # Dense (non-MoR) Phi backbone
      llava_llama.py         # Dense Llama backbone
    mor_modules/
      expert_choice_router.py  # Expert-choice MoR decoder layer + MoRPhiAttention
      token_choice_router.py   # Token-choice MoR decoder layer
      cache_utils.py            # Cache / DynamicCache / RecursiveDynamicCache (KV sharing)
      util.py                    # Router classes (Linear/MLP/WideMLP) + shared output dataclass
    multimodal_encoder/     # Vision tower wrapper (CLIP, etc.)
    multimodal_projector/   # Vision-to-language projection modules
  train/                   # Training entry point (train.py) + custom Trainer
  eval/                    # Per-benchmark evaluation scripts
scripts/
  phi2/                    # 3-stage MoR-Tuning training scripts (pretrain / router warmup / joint finetune)
  eval/mor_mllm/           # Benchmark evaluation launch scripts (GQA, MME, MMBench, ...)
moellava/                  # Vendored LLaVA/MoE-LLaVA infrastructure (see Attribution below)
```

## Training (3-stage MoR-Tuning)

```bash
# Stage 1: projection warmup (align a frozen vision tower to a frozen LLM)
bash scripts/phi2/stage1_projection_warmup.sh

# Stage 2: router warmup (train the MoR router with the backbone frozen)
bash scripts/phi2/stage2_router_warmup.sh

# Stage 3: joint fine-tuning (LoRA) of the full MoR-converted model
bash scripts/phi2/stage3_joint_finetune_lora.sh
```

Each script calls `mor_mllm/train/train.py` via `deepspeed`, with MoR-specific
flags such as `--mor_enable`, `--mor_mode` (`expert` or `token`),
`--sharing_strategy` (`cycle` or `middle_cycle`), `--num_recursions`,
`--router_type`, `--expert_capacity` / `--cap_warmup_step` (expert-choice),
and `--token_balancing` / `--token_router_func` / `--token_alpha` /
`--bal_warmup_step` (token-choice).

## Evaluation

Benchmark launch scripts are under `scripts/eval/mor_mllm/` (GQA, MME, MMBench,
MMVet, SEED, SQA, DocVQA, MathVista, POPE),
each wrapping the corresponding `mor_mllm/eval/*.py` script, which loads a
checkpoint via `moellava.model.builder.load_pretrained_model` and reconstructs
the MoR structure through `EvalMoRLLaVAPhiForCausalLM`.


## Attribution

The code is implemented based on [MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA), and [mixture_of_recursions](https://github.com/raymin0223/mixture_of_recursions). We thank the contributors for their great work!

Licensed under the [Apache License 2.0](LICENSE), consistent with both upstream projects.

