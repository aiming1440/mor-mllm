from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.utils import ModelOutput

def get_torch_dtype(cfg):
    if cfg.precision == "bf16":
        return torch.bfloat16
    elif cfg.precision == "fp16":
        return torch.float16
    elif cfg.precision == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Invalid precision: {cfg.precision}")


class LinearRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, out_dim, bias=False)
        self.router.weight.data.normal_(mean=0.0, std=config.initializer_range)
        
    def forward(self, x):
        return self.router(x)
    
    
class MLPRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, out_dim, bias=False)
        )
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=config.initializer_range)
    
    def forward(self, x):
        return self.router(x)
    
    
class WideMLPRouter(nn.Module):
    def __init__(self, config, out_dim=1):
        super().__init__()
        self.config = config
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, out_dim, bias=False)
        )
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=config.initializer_range)
    
    def forward(self, x):
        return self.router(x)
    
    
ROUTER_TYPES = {
    "linear": LinearRouter, 
    "mlp": MLPRouter, 
    "wide_mlp": WideMLPRouter,
}


@dataclass
class MoRLayerOutputWithPast(ModelOutput):

    hidden_state: Optional[torch.FloatTensor] = None
    attention_weights: Optional[torch.FloatTensor] = None
    selected_tokens: Optional[torch.FloatTensor] = None
    sampling_loss: Optional[torch.FloatTensor] = None
    sampling_acc: Optional[torch.FloatTensor] = None
    sampling_topk_acc: Optional[torch.FloatTensor] = None
    uniformity: Optional[torch.FloatTensor] = None
    dead_token_seq: Optional[torch.FloatTensor] = None
    balancing_loss: Optional[torch.FloatTensor] = None
    balancing_ratio: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None
    # The routing activation strength of each token at the current MoR block position
    # (depth), i.e. sigmoid(router_score) (0 for unselected tokens), shape
    # [bs, seq_len]. Used at the model level to stack the activation strengths from
    # multiple MoR block positions into [bs, seq_len, num_recursions], which after
    # normalization is used to compute the entropy regularization loss.
    depth_activation: Optional[torch.FloatTensor] = None