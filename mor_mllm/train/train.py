"""
MorLLaVA multimodal large language model training script

This script implements the training pipeline for the MorLLaVA model, supporting:
1. Multimodal training data (image+text, video+text)
2. A variety of base language models (Llama, Qwen, Phi, MiniCPM, etc.)
3. LoRA fine-tuning and quantized training
4. MoE (Mixture-of-Experts) architecture
5. Multiple conversation formats and preprocessing schemes
6. DeepSpeed integration and distributed training support

Main components:
- ModelArguments: model-related argument configuration
- DataArguments: data-related argument configuration
- TrainingArguments: training-related argument configuration
- LazySupervisedDataset: lazily-loaded supervised dataset
- DataCollatorForSupervisedDataset: data collator
- train(): main training function
"""

import os
import copy
import random
import sys
from dataclasses import dataclass, field
import json
import logging
import pathlib
from glob import glob
from typing import Dict, Optional, Sequence, List

import torch
import transformers

# Import model constant definitions
from mor_mllm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, MAX_IMAGE_LENGTH, \
    MAX_VIDEO_LENGTH
from torch.utils.data import Dataset
from mor_mllm.train.llava_trainer import LLaVATrainer

# Import conversation-related modules
from mor_mllm import conversation as conversation_lib
from mor_mllm.model import *  # import all model classes
from mor_mllm.mm_utils import tokenizer_image_token

from PIL import Image
from mor_mllm.utils import order_pick_k

# Fix for an error when loading a stage-2 pretrained huggingface model
from mor_mllm.model.language_model.llava_phi import LlavaPhiConfig
from mor_mllm.model.language_model.llava_phi_mor import MoRLLaVAPhiConfig
from transformers import CONFIG_MAPPING, TOKENIZER_MAPPING, CodeGenTokenizer 
CONFIG_MAPPING.register("llava_phi", LlavaPhiConfig)
TOKENIZER_MAPPING.register(LlavaPhiConfig, (CodeGenTokenizer, None))

CONFIG_MAPPING.register("mor_llava_phi", MoRLLaVAPhiConfig)
TOKENIZER_MAPPING.register(MoRLLaVAPhiConfig, (CodeGenTokenizer, None))


# Global variable identifying the rank of the current process
local_rank = None

def rank0_print(*args):
    """Print only on rank 0 to avoid duplicate output during distributed training"""
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    """Model-related argument configuration class

    Contains parameters related to model loading, multimodal components, MoE configuration, etc.
    """
    # ===== Base model configuration =====
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m",
                                              metadata={"help": "Path to the pretrained model or Hugging Face model name"})
    version: Optional[str] = field(default="v0",
                                  metadata={"help": "Conversation template version, affects input formatting"})
    freeze_backbone: bool = field(default=False,
                                 metadata={"help": "Whether to freeze the language model backbone"})

    # ===== Multimodal adapter configuration =====
    tune_mm_mlp_adapter: bool = field(default=False,
                                     metadata={"help": "Whether to fine-tune only the multimodal MLP adapter"})
    mm_vision_select_layer: Optional[int] = field(default=-1,
                                                 metadata={"help": "Which layer of the vision encoder to select features from; -1 means the last layer"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None,
                                                   metadata={"help": "Path to a pretrained multimodal MLP adapter"})
    mm_use_im_start_end: bool = field(default=False,
                                     metadata={"help": "Whether to add start/end markers before and after the image token"})
    mm_use_im_patch_token: bool = field(default=True,
                                       metadata={"help": "Whether to use the image patch token"})
    mm_vision_select_feature: Optional[str] = field(default="patch",
                                                    metadata={"help": "Which vision feature type to select: patch or cls"})

    # ===== Vision tower configuration =====
    image_tower: Optional[str] = field(default=None,
                                      metadata={"help": "Path to the image encoder model"})
    video_tower: Optional[str] = field(default=None,
                                      metadata={"help": "Path to the video encoder model"})
    image_projector_type: Optional[str] = field(default='linear',
                                               metadata={"help": "Image projector type: linear or mlp"})
    video_projector_type: Optional[str] = field(default='linear',
                                               metadata={"help": "Video projector type: linear or mlp"})
    video_global_proj: bool = field(default=False,
                                   metadata={"help": "Whether to use a global video projection"})
    video_temproal_proj: bool = field(default=False,
                                     metadata={"help": "Whether to use a temporal projection"})
    video_spatial_proj: bool = field(default=False,
                                    metadata={"help": "Whether to use a spatial projection"})

    # ===== MoE (Mixture-of-Experts) configuration =====
    only_lora_ffn: bool = field(default=True,
                               metadata={"help": "Whether to apply LoRA only to FFN layers"})
    moe_enable: bool = field(default=False,
                            metadata={"help": "Whether to enable the MoE architecture"})
    train_modules: Optional[List[str]] = field(default=None,
                                              metadata={"help": "List of modules to train"})
    moe_mode: str = field(default="second_half",
                         metadata={"help": "MoE mode: first_half, second_half, sparse, dense",
                                  "choices": ["first_half", "second_half", "sparse", "dense"]})
    moe_layers_idx: Optional[List[int]] = field(default=None,
                                               metadata={"help": "List of layer indices at which to place MoE layers"})
    ep_size: int = field(default=1,
                        metadata={"help": "Expert parallel size"})
    num_experts: Optional[List[int]] = field(default=4,
                                            metadata={"help": "Number of experts per MoE layer"})
    top_k_experts: int = field(default=2,
                              metadata={"help": "Number of top-k experts activated per token",
                                       "choices": [1, 2]})
    capacity_factor: float = field(default=1.,
                                  metadata={"help": "Capacity factor during training"})
    eval_capacity_factor: float = field(default=2.,
                                       metadata={"help": "Capacity factor during evaluation"})
    min_capacity: int = field(default=0,
                             metadata={"help": "Minimum capacity"})
    use_residual: bool = field(default=False,
                              metadata={"help": "Whether to use a residual connection"})
    router_aux_loss_coef: float = field(default=0.01,
                                       metadata={"help": "Router auxiliary loss coefficient"})

    # ===== MoR (Mixture-of-Recursions) configuration =====
    mor_enable: bool = field(default=False,
                            metadata={"help": "Whether to enable the MoR architecture"})
    mor_mode: str = field(default="expert",
                         metadata={"help": "MoR mode: expert or token",
                                  "choices": ["expert", "token"]})
    sharing_strategy: str = field(default="middle_cycle",
                                 metadata={"help": "Sharing strategy: middle_cycle or cycle",
                                          "choices": ["middle_cycle", "cycle"]})
    num_recursions: int = field(default=3,
                               metadata={"help": "Number of recursions / recursion blocks (see MoRLLaVAPhiConfig docstring)"})
    group_size: Optional[int] = field(default=None,
                               metadata={"help": "MoR group size m; only used to derive a default when num_recursions is not explicitly specified"})
    mor_block_init: str = field(default="grouped_sharing",
                               metadata={"help": "Initialization/weight-sharing strategy for the shared submodule M",
                                        "choices": ["grouped_sharing", "mean"]})
    entropy_reg_coeff: float = field(default=0.1,
                               metadata={"help": "Entropy regularization loss coefficient lambda_ent (default 0.1)"})
    freeze_shared_block: bool = field(default=False,
                               metadata={"help": "Stage II (Router Warm-up): freeze the parameters of the MoR shared submodule M"})
    freeze_router: bool = field(default=False,
                               metadata={"help": "Freeze the MoR router parameters (Stage I usually doesn't involve MoR; mainly used for debugging)"})

    rand_router: bool = field(default=False,
                            metadata={"help": "Whether to use random routing"})
    router_type: str = field(default="linear",
                            metadata={"help": "Router type: linear or mlp",
                                     "choices": ["linear", "mlp"]})
    gating: str = field(default="weighted",
                            metadata={"help": "Gating type: weighted or hard",
                                     "choices": ["weighted", "hard"]})

    use_z_loss: bool = field(default=True,
                               metadata={"help": "Whether to use the Z-loss"})
    z_loss_coeff: float = field(default=0.001,
                               metadata={"help": "Z-loss coefficient"})

    kv_sharing_enable: bool = field(default=True,
                                   metadata={"help": "Whether to enable KV sharing"})
    kv_sharing_update_cache: bool = field(default=False,
                                          metadata={"help": "Whether to update the cache during KV sharing"})

    # == Expert-choice-specific configuration ==
    expert_capacity: str = field(default="0.5, 0.3, 0.2",
                          metadata={"help": "Expert capacity configuration"})
    cap_warmup_step: int = field(default=0,
                                 metadata={"help": "Number of expert capacity warmup steps"})
    use_aux_loss: bool = field(default=True,
                               metadata={"help": "Whether to use the auxiliary loss"})
    aux_loss_coeff: float = field(default=0.1,
                                 metadata={"help": "Auxiliary loss coefficient"})

    # == Token-choice-specific configuration ==
    bal_loss_coeff: float = field(default=0.1,
                                 metadata={"help": "Load-balancing loss coefficient"})
    token_balancing: str = field(default="loss",
                                 metadata={"help": "Token-choice balancing mode: 'loss' (auxiliary load-balancing loss) or 'loss_free' (bias-based, no extra loss term)."})
    token_router_func: str = field(default="softmax",
                                 metadata={"help": "Activation function for the token-choice router logits before top-1 selection: 'softmax' or 'sigmoid'."})
    token_alpha: float = field(default=1.0,
                                 metadata={"help": "Scaling factor applied to token-choice router probabilities before top-1 selection and gating."})
    bal_warmup_step: int = field(default=0,
                                 metadata={"help": "Number of initial training steps during which token-choice routing forces every token to the deepest recursion, before the router starts specializing."})

@dataclass
class DataArguments:
    """Data-related argument configuration class

    Contains parameters for data preprocessing, multimodal data configuration, etc.
    """
    # ===== Data preprocessing configuration =====
    lazy_preprocess: bool = field(default=False,
                                 metadata={"help": "Whether to use lazy-loading preprocessing to save memory"})
    is_multimodal: bool = field(default=False,
                               metadata={"help": "Whether the data is multimodal"})
    image_aspect_ratio: str = field(default='square',
                                   metadata={"help": "How to handle image aspect ratio: square or pad"})

    # ===== Data path configuration =====
    data_path: Optional[List[str]] = field(default=None,
                                          metadata={"help": "List of training data file paths"})
    image_folder: Optional[str] = field(default=None,
                                       metadata={"help": "Path to the image data folder"})
    video_folder: Optional[str] = field(default=None,
                                       metadata={"help": "Path to the video data folder"})

    # ===== Video processing configuration =====
    num_frames: int = field(default=8,
                           metadata={"help": "Number of frames sampled per video"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Training-related argument configuration class

    Inherits from transformers.TrainingArguments and adds configuration related to multimodal training, quantization, LoRA, etc.
    """
    # ===== Base training configuration =====
    cache_dir: Optional[str] = field(default=None,
                                    metadata={"help": "Cache directory path"})
    optim: str = field(default="adamw_torch",
                      metadata={"help": "Optimizer type"})
    remove_unused_columns: bool = field(default=False,
                                       metadata={"help": "Whether to remove unused data columns"})
    mpt_attn_impl: Optional[str] = field(default="triton",
                                        metadata={"help": "Attention implementation for the MPT model"})
    model_max_length: int = field(default=512,
                                 metadata={"help": "Maximum model sequence length; sequences will be right-padded or truncated"})

    # ===== Quantization configuration =====
    double_quant: bool = field(default=True,
                              metadata={"help": "Whether to compress quantization statistics via double quantization"})
    quant_type: str = field(default="nf4",
                           metadata={"help": "Quantization data type, either fp4 or nf4"})
    bits: int = field(default=16,
                     metadata={"help": "Number of quantization bits, supports 4/8/16"})

    # ===== LoRA configuration =====
    lora_enable: bool = field(default=False,
                             metadata={"help": "Whether to enable LoRA fine-tuning"})
    lora_r: int = field(default=128,
                       metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=256,
                           metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.05,
                               metadata={"help": "LoRA dropout rate"})
    lora_weight_path: str = field(default="",
                                 metadata={"help": "Path to pretrained LoRA weights"})
    lora_bias: str = field(default="none",
                          metadata={"help": "LoRA bias setting: none/all/lora_only"})

    # ===== Multimodal training configuration =====
    freeze_mm_mlp_adapter: bool = field(default=False,
                                       metadata={"help": "Whether to freeze the multimodal MLP adapter"})
    mm_projector_lr: Optional[float] = field(default=None,
                                            metadata={"help": "Learning rate for the multimodal projector"})
    group_by_modality_length: bool = field(default=False,
                                          metadata={"help": "Whether to group batches by modality length"})



def maybe_zero_3(param, ignore_status=False, name=None):
    """Fetch a parameter's value under DeepSpeed ZeRO-3 optimizer state

    When using DeepSpeed ZeRO-3, parameters may be sharded across different devices.
    This function gathers the sharded parameter and returns a full copy of it.

    Args:
        param: the parameter tensor to process
        ignore_status: whether to skip the parameter status check
        name: parameter name, used for logging

    Returns:
        A full copy of the parameter tensor (on CPU)
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        # Parameter is managed by DeepSpeed
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        # Gather the sharded parameter
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        # Regular (non-sharded) parameter
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    """Get the PEFT (LoRA)-related parameter state dict

    Extracts LoRA-related parameters from the model's named parameters, supporting
    different bias-handling modes. Adapted from peft.utils.get_peft_model_state_dict.

    Args:
        named_params: iterator over the model's named parameters
        bias: bias-handling mode
            - "none": exclude bias
            - "all": include all bias terms
            - "lora_only": include only LoRA-related bias terms

    Returns:
        A dict containing the LoRA parameters
    """
    if bias == "none":
        # Save only LoRA weights, no bias
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        # Save LoRA weights and all bias terms
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        # Save only LoRA weights and their corresponding bias terms
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()

        # Collect LoRA weights and candidate bias terms
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t

        # Keep only bias terms that correspond to a LoRA layer
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError(f"Unsupported bias mode: {bias}")

    # Resolve parameters under DeepSpeed ZeRO-3 state
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """Get the state dict of non-LoRA trainable parameters

    Extracts the model's non-LoRA parameters, typically used to save other
    trainable parameters besides LoRA.

    Args:
        named_params: iterator over the model's named parameters
        require_grad_only: whether to keep only parameters that require gradients

    Returns:
        A dict containing the non-LoRA parameters
    """
    # Filter out LoRA parameters
    to_return = {k: t for k, t in named_params if "lora_" not in k}

    if require_grad_only:
        # Keep only parameters that require gradients
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}

    # Resolve DeepSpeed ZeRO-3 state and move to CPU
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """Get the state dict of multimodal adapter parameters

    Extracts multimodal adapter parameters (e.g. mm_projector) whose names contain
    any of the given keywords.

    Args:
        named_params: iterator over the model's named parameters
        keys_to_match: list of keywords to match against parameter names

    Returns:
        A dict containing the multimodal adapter parameters
    """
    # Filter parameters whose name contains one of the given keywords
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}

    # Resolve DeepSpeed ZeRO-3 state and move to CPU
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model, add_keywords=None):
    """Find the names of all linear layers in the model, for LoRA configuration

    Iterates over all modules of the model and collects the names of torch.nn.Linear
    layers, excluding multimodal-related layers (which normally shouldn't have LoRA
    applied to them).

    Args:
        model: the model to analyze
        add_keywords: additional keywords to exclude

    Returns:
        A list of linear layer names suitable for LoRA
    """
    cls = torch.nn.Linear
    lora_module_names = set()

    # Keywords identifying multimodal-related modules to exclude
    multimodal_keywords = ['mm_projector', 'image_tower', 'video_tower', 'vision_resampler']
    if add_keywords is not None:
        multimodal_keywords.extend(add_keywords)

    # Iterate over all modules
    for name, module in model.named_modules():
        # Skip multimodal-related modules
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue

        # Collect linear layer names
        if isinstance(module, cls):
            names = name.split('.')
            # Take the last component of the name, or the whole name if there's no dot
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Remove lm_head (needed when training in 16-bit precision)
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Safely save the Hugging Face Trainer's model state dict to disk

    This function uses different save strategies depending on the training configuration:
    1. If only the multimodal adapter is being trained, save only the adapter weights
    2. If DeepSpeed is used, use DeepSpeed's save mechanism
    3. Otherwise use the standard save method

    Args:
        trainer: a Hugging Face Trainer instance
        output_dir: output directory path
    """

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only the multimodal MLP adapter is being trained: save only the adapter weights
        keys_to_match = ['mm_projector']  # multimodal projector
        if getattr(trainer.args, "use_im_start_end", False):
            # If image start/end markers are used, also save the related embeddings
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        # Extract multimodal adapter parameters
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        # Save the model config
        trainer.model.config.save_pretrained(output_dir)

        # Determine the save path and filename
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)

        # Save only on the main process, to avoid duplicate saves across processes
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                # Checkpoint save: create a dedicated mm_projector folder
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                # Final save: save directly to the output directory
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        # Use DeepSpeed's model-saving mechanism
        torch.cuda.synchronize()  # synchronize CUDA operations
        trainer.save_model(output_dir)
        return

    # Standard save flow
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        # Move the state dict to CPU to save GPU memory
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict  # free GPU memory
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Smartly resize the tokenizer and the token embedding matrix

    When new special tokens are added, the model's token embedding matrix must be
    resized accordingly. Newly added embeddings are initialized to the average of
    the existing embeddings.

    Note: this is the unoptimized version — it may result in an embedding size
    that is not a multiple of 64.

    Args:
        special_tokens_dict: dict containing the special tokens
        tokenizer: the pretrained tokenizer
        model: the pretrained model
    """
    # Add special tokens to the tokenizer, returning the number of newly added tokens
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # Resize the model's token embedding matrix to match the new vocabulary size
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        # Get the input and output embedding weights
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # Compute the average of the existing embeddings
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        # Initialize the newly added embeddings with the average
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings

    Tokenizes the input string sequence with the given tokenizer and returns a dict
    containing input_ids, labels, and their corresponding length information.

    Args:
        strings: sequence of strings to tokenize
        tokenizer: the pretrained tokenizer

    Returns:
        A dict containing the tokenization results and length information
    """
    # Tokenize each string
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",           # return PyTorch tensors
            padding="longest",             # pad to the longest sequence length
            max_length=tokenizer.model_max_length,  # maximum sequence length
            truncation=True,               # truncate if too long
        ) for text in strings
    ]

    # Extract input_ids, used for both model input and labels
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]

    # Compute the actual length of each sequence (excluding padding tokens)
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    """Mask out the portions of the target sequence that should not contribute to the loss

    In supervised fine-tuning we only compute loss on the assistant's (GPT) replies;
    the system prompt and human-input portions are masked out with IGNORE_INDEX.

    Args:
        target: the target sequence tensor, modified in place
        tokenized_lens: list of tokenized lengths for each segment
        speakers: list of speakers ('human' or 'gpt')
    """
    # Start right after the system prompt
    cur_idx = tokenized_lens[0]  # skip the system prompt segment
    tokenized_lens = tokenized_lens[1:]  # remove the system prompt length

    # Mask out the system prompt segment
    target[:cur_idx] = IGNORE_INDEX

    # Iterate over each conversation turn
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            # Mask out the human-input segment (+2 likely skips some special tokens)
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        # GPT's reply is left unmasked, so the model learns to generate this content
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add a speaker identifier and begin/end signal to each conversation turn

    Formats the conversation data with an explicit speaker identifier so the model
    can recognize inputs from different roles and process them accordingly.

    Args:
        header: the opening portion of the conversation (usually the system prompt)
        source: list of conversation data, each element containing 'from' and 'value' keys
        get_conversation: whether to return the full conversation string

    Returns:
        The formatted conversation string
    """
    BEGIN_SIGNAL = "### "    # begin signal
    END_SIGNAL = "\n"       # end signal
    conversation = header

    for sentence in source:
        from_str = sentence["from"]

        # Map the speaker type to the corresponding role name
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]  # usually "Human" or "User"
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]  # usually "Assistant" or "GPT"
        else:
            from_str = 'unknown'  # unknown role

        # Add formatting markers to each utterance
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)

        if get_conversation:
            conversation += sentence["value"]

    conversation += BEGIN_SIGNAL  # append the begin signal at the end, in preparation for the next turn
    return conversation



def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:

            # ======================================================================================================
            if sentence['value'].startswith(DEFAULT_IMAGE_TOKEN) or sentence['value'].startswith(DEFAULT_VIDEO_TOKEN):  # run with multi-im, multi-vid, multi-im & multi-vid
                # <video><video><image><image>\nxxxxxxxxxxxxx  # must <video> first
                # <image>\nxxxxxxxxxxxxx -> <image>\nxxxxxxxxxxxxx
                # <video>\nxxxxxxxxxxxxx -> <video>\nxxxxxxxxxxxxx

                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')

                IMAGE_TOKEN_NUM = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                if IMAGE_TOKEN_NUM > MAX_IMAGE_LENGTH:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM, DEFAULT_IMAGE_TOKEN * MAX_IMAGE_LENGTH).strip()
                VIDEO_TOKEN_NUM = sentence['value'].count(DEFAULT_VIDEO_TOKEN)
                if VIDEO_TOKEN_NUM > MAX_VIDEO_LENGTH:
                    raise ValueError(f"{sentence['value']}")
                    sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM, DEFAULT_VIDEO_TOKEN * MAX_VIDEO_LENGTH).strip()

            # a <video> is treated as `num_frames * <image>`
            replace_token, vid_replace_token = DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN * data_args.num_frames
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                vid_replace_token = DEFAULT_VID_START_TOKEN + vid_replace_token + DEFAULT_VID_END_TOKEN

            # <video><video><image><image>\nxxxxxxxxxxxxx -> `num_frames*<image>``num_frames*<image>`<image><image>\nxxxxxxxxxxxxx
            # <video>\nxxxxxxxxxxxxx -> `num_frames*<image>`\nxxxxxxxxxxxxx
            # print('before replace_token:', [sentence['value']])
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, vid_replace_token)
            # print('after replace_token:', [sentence['value']])
            # ======================================================================================================

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # print('00000000000', sources)
    # Apply prompt templates
    conversations = []
    # sys.exit()

    # import ipdb
    # ipdb.set_trace()
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(11111111, conversations)
    # Tokenize conversations
    # print('before tokenizer_image_token', conversations)
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        # print(2222222222222, input_ids.shape)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # print('after tokenizer_image_token', input_ids)
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # print(tokenizer)
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # print('total_len', total_len)
        rounds = conversation.split(conv.sep2)
        # print('len(rounds)', len(rounds))
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            # import ipdb
            # ipdb.set_trace()
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_phi(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # print('00000000000', sources)
    # Apply prompt templates
    conversations = []
    # sys.exit()

    # import ipdb
    # ipdb.set_trace()
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(11111111, conversations)
    # Tokenize conversations
    # print('before tokenizer_image_token', conversations)
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        # print(2222222222222, input_ids.shape)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # print('after tokenizer_image_token input_ids targets', input_ids)
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # print(tokenizer)
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    # print('sep', sep)
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # print('total_len', total_len)
        rounds = conversation.split(conv.sep2)
        # print('len(rounds)', len(rounds))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # print('i rou, parts', i, rou, parts)
            if len(parts) != 2:
                break
            parts[0] += sep
            # print('after add sep, parts', parts)

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # for eos_token
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) + 1  # for eos_token
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            # print('round_len, instruction_len, target[cur_len : cur_len + instruction_len]',
            #       round_len, instruction_len, target[cur_len : cur_len + instruction_len], target[cur_len : cur_len + round_len])
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  # instruction_len is before the answer

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            # import ipdb
            # ipdb.set_trace()
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    # print(input_ids, target)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_openchat(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # print('00000000000', sources)
    # Apply prompt templates
    conversations = []
    # sys.exit()

    # import ipdb
    # ipdb.set_trace()
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(11111111, conversations)
    # Tokenize conversations
    # print('before tokenizer_image_token', conversations)
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        # print(2222222222222, input_ids.shape)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # print('after tokenizer_image_token input_ids targets', input_ids)
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # print(tokenizer)
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    # print('sep\n', sep)
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # print('total_len', total_len)
        rounds = conversation.split(conv.sep2)
        # print('len(rounds)', len(rounds))
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # print('i rou, parts\n', i, rou, parts)
            if len(parts) != 2:
                break
            parts[0] += sep
            # print('after add sep, parts\n', parts)

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            # print('instruction_len, target[cur_len : cur_len + instruction_len]\n',
            #       instruction_len, target[cur_len : cur_len + instruction_len])
            # print('round_len, target[cur_len : cur_len + round_len]\n',
            #       round_len, target[cur_len : cur_len + round_len])
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  # instruction_len is before the answer

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        # print(cur_len, total_len)
        if cur_len < tokenizer.model_max_length:
            # import ipdb
            # ipdb.set_trace()
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    # print(input_ids, target)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    # print('sources', sources)
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # print('conversations', conversations)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    # print('after tokenizer_image_token', input_ids)
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    # print('target:', target)
    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("phi") or \
            conversation_lib.default_conversation.version.startswith("qwen"):  # for phi and qwen
        return preprocess_phi(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("stablelm"):  # stablelm same as phi
        return preprocess_phi(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("openchat") or \
        conversation_lib.default_conversation.version.startswith("mistral"):  # for openchat
        return preprocess_openchat(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("minicpm"):  # minicpm same as openchat
        return preprocess_openchat(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)



def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class LazySupervisedDataset(Dataset):
    """Multimodal supervised learning dataset (lazy-loading mode)

    This dataset supports image-text, video-text, and mixed-modality training data.
    It uses lazy loading, i.e. individual samples are only loaded and processed when
    needed, to save memory.

    Supported data types:
    - Text-only conversations
    - Image+text conversations
    - Video+text conversations
    - Image+video+text conversations
    """

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        """Initialize the dataset

        Args:
            data_path: list of data file paths
            tokenizer: the pretrained tokenizer
            data_args: data-related argument configuration
        """
        super(LazySupervisedDataset, self).__init__()

        # Load and merge all data files
        list_data_dict = []
        for data in data_path:
            data = json.load(open(data, "r"))
            for i in data:
                i['id'] = len(list_data_dict)  # assign a unique ID to each sample
                list_data_dict.append(i)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        """Compute a per-sample modality length, used for batch grouping

        To improve training efficiency, samples are grouped by modality type and
        length:
        - Multimodal samples (containing image or video): return a positive length
        - Text-only samples: return a negative length

        This lets samples of similar type and length be placed in the same batch,
        reducing padding and improving training throughput.

        Returns:
            A list of per-sample lengths, positive for multimodal, negative for text-only
        """
        length_list = []
        for sample in self.list_data_dict:
            # Compute the total length of the conversation content (simple whitespace split)
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])

            # Positive for multimodal samples, negative for text-only, to enable grouping
            cur_len = cur_len if ('image' in sample or 'video' in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            # ======================================================================================================
            if 'image' in sources[0] and 'video' not in sources[0]:
                # rank0_print('image')
                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                image_processor = self.data_args.image_processor
                image_file = image_file if isinstance(image_file, list) else [image_file]
                image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
                # print(f"total {len(self.list_data_dict[i]['image'])} now {len(image_file)}")
                image = [Image.open(os.path.join(image_folder, file)).convert('RGB') for file in image_file]
                # print(image[0])
                if self.data_args.image_aspect_ratio == 'pad':
                    image = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in image]
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                else:
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                # print(image[0].shape)
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                data_dict = preprocess(sources, self.tokenizer, has_image=True)

            elif 'image' not in sources[0] and 'video' in sources[0]:
                # rank0_print('video')
                video_file = self.list_data_dict[i]['video']
                video_folder = self.data_args.video_folder
                video_processor = self.data_args.video_processor
                video_file = video_file if isinstance(video_file, list) else [video_file]
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                image = [video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video]  # fake image
                # image = [torch.randn(3, 8, 224, 224) for i in video]  # fake image
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                # print('after preprocess_multimodal', sources[0])
                data_dict = preprocess(sources, self.tokenizer, has_image=True)
                # print('after preprocess', data_dict['input_ids'])

            elif 'image' in sources[0] and 'video' in sources[0]:
                # rank0_print('image & video')
                # video must before image
                video_file = self.list_data_dict[i]['video']
                video_folder = self.data_args.video_folder
                video_processor = self.data_args.video_processor

                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                image_processor = self.data_args.image_processor

                image_file = image_file if isinstance(image_file, list) else [image_file]
                image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
                image = [Image.open(os.path.join(image_folder, file)).convert('RGB') for file in image_file]
                if self.data_args.image_aspect_ratio == 'pad':
                    image = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in image]
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                else:
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]

                video_file = video_file if isinstance(video_file, list) else [video_file]
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                video = [video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video]  # fake image

                image = video + image  # video must before image

                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                data_dict = preprocess(sources, self.tokenizer, has_image=True)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess(sources, self.tokenizer, has_image=False)

            # ==========================================================================================================

            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0])

            # --- [Core fix] Truncate the single sample here ---
            # Whether the data is multimodal or text-only, handle length uniformly at this point
            max_len = self.tokenizer.model_max_length
            data_dict['input_ids'] = data_dict['input_ids'][:max_len]
            data_dict['labels'] = data_dict['labels'][:max_len]
            # -------------------------------------------------

            # image exist in the data
            if 'image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i]:
                data_dict['image'] = image
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                if hasattr(self.data_args.image_processor, 'crop_size'):
                    crop_size = self.data_args.image_processor.crop_size
                    data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]
                else:
                    size = self.data_args.image_processor.size
                    data_dict['image'] = [torch.zeros(3, size['height'], size['width'])]
            return data_dict
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__()-1))


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # print('before Collator', input_ids)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # print('after Collator', batch)
        # print(input_ids, labels, input_ids.ne(self.tokenizer.pad_token_id))
        # ======================================================================================================
        # origin image, if batch_size=6: [[image], [image], [video], [image, image], [video, video], [video, image]]
        '''
            will be converted to a sequence of list, if batch size=6:
            [
                image(3, 224, 224),      # sample 1
                image(3, 224, 224),      # sample 2
                video(8, 3, 224, 224),   # sample 3
                image(3, 224, 224),      # sample 4
                image(3, 224, 224),      # sample 4
                video(8, 3, 224, 224),   # sample 5
                video(8, 3, 224, 224),   # sample 5
                video(8, 3, 224, 224),   # sample 6
                image(3, 224, 224),      # sample 6
            ]
        '''
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]

            # adapt to multi-video or multi-image or multi-image & video
            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images

        # ==========Too many videos or images may lead to OOM, so we encode them one by one======================
            batch['images'] = images
        #     if all(x is not None and x.shape == images[0].shape for x in images):  # if all images or all videos
        #         batch['images'] = torch.stack(images)
        #     else:
        #         batch['images'] = images
        else:
            raise ValueError(f'pretrain, {instances}')
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    """Main training function for the MorLLaVA model

    This function implements the full multimodal large language model training
    pipeline, including:
    1. Argument parsing and environment setup
    2. Quantization configuration (supports 4-bit and 8-bit quantization)
    3. Model loading (supports various base models and the MoE architecture)
    4. LoRA configuration (optional)
    5. Tokenizer initialization
    6. Multimodal component initialization
    7. Dataset preparation and training execution
    8. Model saving
    """
    global local_rank

    # ===== 1. Argument parsing and base configuration =====
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Determine the compute dtype from the training arguments
    compute_dtype = (torch.float16 if training_args.fp16 else
                    (torch.bfloat16 if training_args.bf16 else torch.float32))

    # ===== 2. Quantization configuration =====
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        # Configure BitsAndBytes quantization parameters
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],  # skip quantization for the multimodal projector
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # 'fp4' or 'nf4'
            )
        ))

    # ===== 3. Model loading =====
    if model_args.image_tower is not None or model_args.video_tower is not None:
        # Multimodal model loading
        if model_args.mor_enable:
            # MoR architecture model loading.
            if 'phi' in model_args.model_name_or_path.lower():
                model = MoRLLaVAPhiForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            else:
                raise NotImplementedError(
                    f"MoR-MLLM currently only implements the MoR version of the Phi backbone "
                    f"(MoRLLaVAPhiForCausalLM); the backbone for "
                    f"model_name_or_path='{model_args.model_name_or_path}' is not supported."
                )
        elif not model_args.moe_enable:
            # Standard multimodal model (non-MoE)
            if 'mpt' in model_args.model_name_or_path.lower():
                config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
                config.attn_config['attn_impl'] = training_args.mpt_attn_impl
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
            elif 'qwen' in model_args.model_name_or_path.lower() and '1.5' not in model_args.model_name_or_path.lower():
                model = LlavaQWenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
            elif 'qwen' in model_args.model_name_or_path.lower() and '1.5' in model_args.model_name_or_path.lower():
                model = LlavaQwen1_5ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'openchat' in model_args.model_name_or_path.lower() or 'mistral' in model_args.model_name_or_path.lower():
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'phi' in model_args.model_name_or_path.lower():
                model = LlavaPhiForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'minicpm' in model_args.model_name_or_path.lower():
                model = LlavaMiniCPMForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'stablelm' in model_args.model_name_or_path.lower():
                model = LlavaStablelmForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
        else:
            if 'qwen' in model_args.model_name_or_path.lower() and '1.5' not in model_args.model_name_or_path.lower():
                model = MoELLaVAQWenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
            elif 'qwen' in model_args.model_name_or_path.lower() and '1.5' in model_args.model_name_or_path.lower():
                model = MoELLaVAQwen1_5ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'phi' in model_args.model_name_or_path.lower():
                model = MoELLaVAPhiForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'minicpm' in model_args.model_name_or_path.lower():
                model = MoELLaVAMiniCPMForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'openchat' in model_args.model_name_or_path.lower() or 'mistral' in model_args.model_name_or_path.lower():
                model = MoELLaVAMistralForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            elif 'stablelm' in model_args.model_name_or_path.lower():
                model = MoELLaVAStablelmForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    # attn_implementation="flash_attention_2",
                    # torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
            else:
                model = MoELLaVALlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            # attn_implementation="flash_attention_2",
            # torch_dtype=torch.bfloat16,
            **bnb_model_from_pretrained_args
        )
    rank0_print('LLM init. firstly\n', model)
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # ==============================================================================================
    training_args.moe_enable = model_args.moe_enable
    training_args.only_lora_ffn = model_args.only_lora_ffn
    model_args.lora_enable = training_args.lora_enable
    if model_args.moe_enable:
        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            if 'qwen' in model_args.model_name_or_path.lower() and '1.5' not in model_args.model_name_or_path.lower():
                target_modules = [
                    'mlp.w1', 'mlp.w2', 'mlp.c_proj'
                ] if training_args.only_lora_ffn else find_all_linear_names(model)
            elif 'phi' in model_args.model_name_or_path.lower():
                target_modules = [
                    'fc1', 'fc2'
                ] if training_args.only_lora_ffn else find_all_linear_names(model)
            else:
                target_modules = [
                    'up_proj', 'down_proj', 'gate_proj'
                ] if training_args.only_lora_ffn else find_all_linear_names(model)
            # modules_to_save = ['wg']  # weight gating for MoE
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                # modules_to_save=modules_to_save,
                task_type="CAUSAL_LM",
            )
            model_args.lora_r = training_args.lora_r
            model_args.lora_alpha = training_args.lora_alpha
            model_args.lora_dropout = training_args.lora_dropout
            model_args.lora_bias = training_args.lora_bias
            # model_args.modules_to_save = modules_to_save
            model_args.target_modules = target_modules
            model_args.train_modules = target_modules
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
        model.initialize_moe_modules(model_args=model_args)
    elif model_args.mor_enable:
        model.initialize_mor_modules(model_args=model_args)

        # ===== MoR-Tuning Stage I/II staged-freezing logic =====
        # Stage II (Router Warm-up): freeze the shared submodule M and other backbone
        # parameters, training only the router + projector. Enabled via
        # --freeze_shared_block True; whether a parameter "belongs to the shared
        # submodule" is determined by checking if its name (from named_parameters)
        # contains '.block.' (MoRPhiDecoderLayer stores the shared layer in self.block).
        if getattr(model_args, 'freeze_shared_block', False):
            for n, p in model.named_parameters():
                if '.block.' in n:
                    p.requires_grad = False
            rank0_print("Stage II: MoR shared submodule M parameters frozen (training only router + projector).")

        # Freezing the router alone is only for debugging/special cases (not used in
        # the normal three-stage pipeline).
        if getattr(model_args, 'freeze_router', False):
            for n, p in model.named_parameters():
                if '.router.' in n:
                    p.requires_grad = False
            rank0_print("MoR router parameters frozen.")

        # Stage III (Joint Fine-tuning): unfreeze, then apply LoRA on top.
        if training_args.lora_enable:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as e:
                raise ImportError(
                    "--lora_enable True was passed, but peft cannot be imported in the "
                    f"current environment (original error: {e}). MoR Stage III (Joint "
                    "Fine-tuning + LoRA) depends on peft; please run this stage on a "
                    "machine with matching peft/accelerate versions. The MoR+LoRA code "
                    "path in this repo is implemented, but its runtime behavior could "
                    "not be verified in the current sandbox environment."
                ) from e

            # find_all_linear_names scans the model for all nn.Linear names (excluding
            # multimodal modules like mm_projector). Under the MoR architecture this
            # naturally covers the router's linear layer (LinearRouter.router) as well
            # as the attention/MLP projection layers inside the shared submodule
            # self.block, with no need to hardcode extra module names. Since self.block
            # at all num_recursions MoR-block positions points to the same physical
            # object, peft's traversal via model.named_modules() automatically
            # deduplicates it (an nn.Module reached via multiple reference paths to the
            # same submodule is only visited once), so the LoRA adapter is naturally
            # shared across all recursion positions too, without breaking weight tying.
            target_modules = find_all_linear_names(model)
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            model_args.lora_r = training_args.lora_r
            model_args.lora_alpha = training_args.lora_alpha
            model_args.lora_dropout = training_args.lora_dropout
            model_args.lora_bias = training_args.lora_bias
            model_args.target_modules = target_modules

            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters to MoR model (Stage III)...")
            model = get_peft_model(model, lora_config)
    else:
        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
    # ==============================================================================================

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        # import ipdb
        # ipdb.set_trace()
        if 'qwen' in model_args.model_name_or_path.lower() and '1.5' not in model_args.model_name_or_path.lower():
            from mor_mllm.model.language_model.qwen.tokenization_qwen import QWenTokenizer
            tokenizer = QWenTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.add_special_tokens({'unk_token': '<|extra_0|>', 'eos_token': '<|endoftext|>'})
        if 'qwen' in model_args.model_name_or_path.lower() and '1.5' in model_args.model_name_or_path.lower():
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'})
        elif 'phi' in model_args.model_name_or_path.lower():
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'})
        elif 'stablelm' in model_args.model_name_or_path.lower():
            from mor_mllm.model.language_model.stablelm.tokenization_arcade100k import Arcade100kTokenizer
            tokenizer = Arcade100kTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.unk_token = '<|reg0|>'  # FIXME: DO SUPPORT ADD SPECIAL TOKENS
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
    # import ipdb
    # ipdb.set_trace()
    # print(tokenizer)
    # print(tokenizer)
    # =============================================================================================================
    # Configure the tokenizer's pad_token
    # =============================================================================================================
    if model_args.version == "v0":
        # v0: if the tokenizer has no pad_token, add "[PAD]" as the pad_token
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        # v0.5: use unk_token as the pad_token
        tokenizer.pad_token = tokenizer.unk_token
    else:
        # Other versions: use unk_token as the pad_token, and configure the model
        tokenizer.pad_token = tokenizer.unk_token
        # =============================================================================================================
        # Sync the tokenizer's pad_token_id into the model config
        model.config.pad_token_id = tokenizer.pad_token_id
        # =============================================================================================================
        # Set the conversation template
        if model_args.version in conversation_lib.conv_templates:
            # If the specified version exists among the conversation templates, use it
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            # Otherwise fall back to the default vicuna_v1 conversation template
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    # print(conversation_lib.default_conversation)

    # =============================================================================================================
    # Initialize vision modules (image tower and video tower)
    # =============================================================================================================
    if model_args.image_tower is not None or model_args.video_tower is not None:
        # print(model_args)
        # Initialize vision modules
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        # Initialize the image tower
        if model_args.image_tower is not None:
            image_tower = model.get_image_tower()
            # Move the image tower to the target dtype and device
            image_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            # Set data arguments
            data_args.image_processor = image_tower.image_processor
            data_args.is_multimodal = True

        # Initialize the video tower
        if model_args.video_tower is not None:
            video_tower = model.get_video_tower()
            # Move the video tower to the target dtype and device
            video_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            # Set data arguments
            data_args.video_processor = video_tower.video_processor
            data_args.is_multimodal = True
            data_args.num_frames = video_tower.config.num_frames
    # =============================================================================================================

        # Configure model parameters
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        # model.config.tokenizer_model_max_length = tokenizer.model_max_length  # video token count may exceed 2048

        # Configure multimodal MLP adapter fine-tuning
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            # If only fine-tuning the multimodal MLP adapter, freeze everything else
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        # Configure multimodal MLP adapter freezing
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            # If freezing the multimodal MLP adapter, disable gradients for its parameters
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        # During quantized training, cast the multimodal projector to the target dtype
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        # Configure multimodal-related parameters
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

        # Initialize the vision tokenizer
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    rank0_print('Vision encoder and proj init.\n', model)
    # During quantized training, cast LoRA layers to the target dtype
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        # Iterate over all modules in the model
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    # If using bf16, cast the module to bfloat16
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                # If the module name contains 'norm', cast it to float32
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    # If the module has a weight and bf16 is used, cast it to bfloat16
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    # Iterate over all model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)
    rank0_print(model)
    # sys.exit()

    # Prepare the data module
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    # Create the LLaVA trainer instance
    # Uses the custom LLaVATrainer class, which handles multimodal data specially
    trainer = LLaVATrainer(model=model,
                    processing_class=tokenizer,  # pass in the tokenizer as the processing class
                    args=training_args,          # pass in the training arguments
                    **data_module)               # unpack the data module (contains train_dataset, etc.)

    # Check whether a checkpoint file exists in the output directory, to decide whether to resume training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # If a checkpoint exists, resume training from it
        trainer.train(resume_from_checkpoint=True)
    else:
        # Otherwise, start training from scratch
        trainer.train()

    # Save the trainer state (optimizer state, LR scheduler state, etc.)
    trainer.save_state()

    # After training, re-enable the model's cache mechanism to speed up inference
    model.config.use_cache = True

    # Save the model according to the training configuration
    if training_args.lora_enable and not model_args.moe_enable:
        # If LoRA is enabled but MoE is not, use the LoRA-specific save path

        # Get the LoRA-related state dict, with DeepSpeed ZeRO-3 support
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        # Get the state dict of non-LoRA parameters
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )

        # Save only on the main process (rank 0) or in single-GPU training
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            # Save the model config
            model.config.save_pretrained(training_args.output_dir)
            # Save the LoRA adapter weights
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            # Save the non-LoRA trainable parameters (e.g. the multimodal projector)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        # If LoRA is not enabled, or MoE is enabled, use the standard save path

        # Use the Hugging Face trainer's safe-save method
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

        if model_args.moe_enable:
            # If MoE is enabled, the model state dict needs special handling

            # Get the full model state dict
            ckpt = model.state_dict()

            # Clean up key names in the state dict, removing prefixes added by the PEFT wrapper
            # Remove the 'base_model.' prefix (added by PEFT)
            ckpt = {(k[11:] if k.startswith('base_model.') else k): v for k, v in ckpt.items()}

            # If a 'model.model.' prefix exists, remove the redundant 'model.' prefix
            if any(k.startswith('model.model.') for k in ckpt):
                ckpt = {(k[6:] if k.startswith('model.') else k): v for k, v in ckpt.items()}

            # Save the cleaned-up model weights
            torch.save(ckpt, os.path.join(training_args.output_dir, 'pytorch_model.bin'))

            # Save the model config
            model.config.save_pretrained(training_args.output_dir)

            # Clean up unneeded adapter files only on the main process
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                # Remove all adapter-related files, since the MoE model doesn't need them
                [os.remove(i) for i in glob(os.path.join(training_args.output_dir, 'adapter_*'))]

if __name__ == "__main__":
    train()
