#    Copyright 2024 [Your Name], based on Haotian Liu's MoE-LLaVA
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

# --- MoE-LLaVA Imports ---
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

# --- MoR Imports (Assume these files are copied to mor_modules/) ---
# You need to copy the relevant files from the MoR repository first.
# Make sure the import paths are correct.
from ..mor_modules.expert_choice_router import MoRLlamaDecoderLayer
from ..mor_modules.cache_utils import RecursiveDynamicCache
from ..mor_modules.util import MoRLayerOutputWithPast, ROUTER_TYPES

logger = logging.get_logger(__name__)

# --- 1. Define a new Config class for MoR-LLaVA ---
class MoRLlavaLlamaConfig(LlamaConfig):
    model_type = "mor_llava_llama"

    # You can add MoR specific configurations here, similar to MoE-LLaVA
    # For simplicity, we will pass them via a config object (cfg) later.


# --- 2. Define the MoR version of LlamaModel ---
# This class will have its forward method monkey-patched later.
class MoRLlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = MoRLlavaLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(MoRLlavaLlamaModel, self).__init__(config)


# --- 3. Define the main CausalLM Model Class ---
class MoRLlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = MoRLlavaLlamaConfig

    def __init__(self, config):
        # Initialize as a standard LlamaForCausalLM first
        super(LlamaForCausalLM, self).__init__(config)
        # Replace the standard LlamaModel with our MoR-capable LlamaModel
        self.model = MoRLlavaLlamaModel(config)
        
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        # Placeholder for MoR auxiliary loss coefficient
        self.mor_aux_loss_coef = 0.01 

    def get_model(self):
        return self.model

    # --- 4. The core transformation logic from MoR repository ---
    def transform_layer_to_mor_expert(self, mor_cfg):
        """
        Dynamically rewrites the model's layers to implement the MoR architecture.
        This function is adapted directly from the MoR repository.
        """
        logger.info("Transforming model layers to MoR (Expert Choice) architecture...")
        
        # Configuration from your training script/config file
        sharing = mor_cfg.recursive.sharing
        num_recursion = mor_cfg.recursive.num_recursion
        capacity = [float(cap) for cap in mor_cfg.mor.capacity.split(',')]
        
        num_hidden_layers = len(self.model.layers)

        if sharing == "middle_cycle":
            if num_hidden_layers - 2 < num_recursion:
                raise ValueError("Not enough layers for middle_cycle sharing.")
            
            base_depth = (num_hidden_layers - 2) // num_recursion
            
            # Create the new list of layers
            new_layers = nn.ModuleList(
                [self.model.layers[0]] +  # Entry layer
                [
                    MoRLlamaDecoderLayer(
                        self.config, 
                        nn.ModuleList([self.model.layers[1 + layer_idx + recur_idx * base_depth] for layer_idx in range(base_depth)]),
                        mor_cfg, 
                        capacity[recur_idx]
                    )
                    for recur_idx in range(num_recursion)
                ] +
                [self.model.layers[-1]]  # Exit layer
            )
            self.model.layers = new_layers
            logger.info(f"Reconstructed model with Middle-Cycle MoR. Structure: 1 (Entry) + {num_recursion} (MoR Blocks of depth {base_depth}) + 1 (Exit).")
        else:
            raise NotImplementedError(f"Sharing strategy '{sharing}' is not implemented for this script.")

    def set_kv_sharing_config(self, mor_cfg):
        """
        Configures the model for Recursive KV Sharing.
        Adapted from the MoR repository.
        """
        if mor_cfg.kv_sharing.enable:
            logger.info("Enabling Recursive KV Sharing.")
            if mor_cfg.kv_sharing.sharing in ["middle_cycle"]:
                base_depth = (self.config.num_hidden_layers - 2) // mor_cfg.kv_sharing.num_recursion
            else:
                raise NotImplementedError("Only middle_cycle KV sharing is supported.")
            
            kv_kwargs = {
                "base_depth": base_depth,
                "num_recursion": mor_cfg.kv_sharing.num_recursion,
                "sharing": mor_cfg.kv_sharing.sharing,
                "update_cache": mor_cfg.kv_sharing.get("update_cache", False),
            }
            self.model.config.kv_sharing = kv_kwargs
        else:
            self.model.config.kv_sharing = None

    # --- 5. The Monkey Patching entry point, inspired by MoE-LLaVA ---
    def initialize_mor_modules(self, mor_cfg):
        """
        This is the main function to call after model creation to convert it.
        """
        # Step 1: Reconstruct the model layers into MoR structure
        # Currently only supports expert choice
        if mor_cfg.mor.type == "expert":
            self.transform_layer_to_mor_expert(mor_cfg)
        else:
            raise NotImplementedError("Only expert-choice MoR is currently supported.")
        
        # Step 2: Configure KV Caching
        self.set_kv_sharing_config(mor_cfg)
        
        # Step 3: Monkey-patch the forward pass of the LlamaModel
        self.model.forward = MoRLlamaModel_forward(self.model)
        logger.info("Replaced LlamaModel.forward with MoR-compatible forward pass.")

        # Store aux loss coefficient
        self.mor_aux_loss_coef = mor_cfg.mor.get("aux_loss_coef", 0.01)

    # --- 6. The main forward pass, adapted from MoE-LLaVA ---
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # This part is the core logic from LLaVA for handling multimodal inputs
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        # Call the potentially monkey-patched model.forward
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
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Standard Causal LM loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # --- Handle MoR Auxiliary Losses ---
        aux_loss = 0.0
        # `outputs` is a custom ModelOutput from our monkey-patched forward
        if hasattr(outputs, 'sampling_loss') and outputs.sampling_loss is not None:
            aux_loss += outputs.sampling_loss
        if hasattr(outputs, 'router_z_loss') and outputs.router_z_loss is not None:
            aux_loss += outputs.router_z_loss
        
        if loss is not None and aux_loss > 0:
            loss += self.mor_aux_loss_coef * aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # Use the standard CausalLMOutputWithPast for compatibility,
        # but you can create a custom MoRCausalLMOutput for more detailed logging.
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


# --- 7. The Monkey-Patchable forward function for LlamaModel ---
# This function is adapted from MoR's MoRLlamaModel.forward
def MoRLlamaModel_forward(self):
    def forward(
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # This is a simplified version focusing on the core logic.
        # The original MoR code has more complex handling for cache_position, etc.
        # which can be added back as needed.
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # KV Cache setup
        if use_cache and past_key_values is None:
            if hasattr(self.config, "kv_sharing") and self.config.kv_sharing is not None:
                kv_kwargs = self.config.kv_sharing
                past_key_values = RecursiveDynamicCache(kv_kwargs["base_depth"], kv_kwargs["num_recursion"], kv_kwargs["sharing"], kv_kwargs["update_cache"])
            else:
                # Fallback to standard cache if MoR is not fully configured
                from transformers.cache_utils import DynamicCache
                past_key_values = DynamicCache()

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        # --- MoR specific state ---
        prev_selected_tokens = None
        sampling_loss = torch.tensor(0.0, device=hidden_states.device)
        router_z_loss = torch.tensor(0.0, device=hidden_states.device)
        
        # --- The main recursive loop (implemented as a sequential pass) ---
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_args = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_value": past_key_values,
                "output_attentions": output_attentions,
                "use_cache": use_cache,
            }

            if hasattr(decoder_layer, "mor") and decoder_layer.mor:
                layer_args["prev_selected_tokens"] = prev_selected_tokens
                layer_outputs = decoder_layer(**layer_args)
                
                # Pass the baton
                prev_selected_tokens = layer_outputs.selected_tokens
                
                # Collect auxiliary losses
                if layer_outputs.sampling_loss is not None:
                    sampling_loss += layer_outputs.sampling_loss
                if layer_outputs.router_z_loss is not None:
                    router_z_loss += layer_outputs.router_z_loss
            else:
                # For entry/exit layers
                layer_outputs = decoder_layer(**layer_args)

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Create a custom output object to carry the aux losses
        # This is a simplified version of MoR's output class
        @dataclass
        class MoRModelOutput(CausalLMOutputWithPast):
            sampling_loss: torch.FloatTensor = None
            router_z_loss: torch.FloatTensor = None

        if not return_dict:
            return (hidden_states,) + (past_key_values,) + all_hidden_states + all_self_attns

        return MoRModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            sampling_loss=sampling_loss,
            router_z_loss=router_z_loss,
        )
    return forward


# --- 8. Register the new model with AutoClasses ---
AutoConfig.register("mor_llava_llama", MoRLlavaLlamaConfig)
AutoModelForCausalLM.register(MoRLlavaLlamaConfig, MoRLlavaLlamaForCausalLM)