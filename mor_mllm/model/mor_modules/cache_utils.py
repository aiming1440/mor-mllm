from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
# from transformers.cache_utils import _static_cache_update

logger = logging.get_logger(__name__)


class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    is_compileable = False

    def __init__(self):
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        raise NotImplementedError("Make sure to implement `get_max_cache_shape` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] != []:
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx] != []:
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` "
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    def __init__(self) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.value_cache.append([])
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                len(self.key_cache[layer_idx]) == 0
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx] != []:
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["DynamicCache"]) -> "DynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            key_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx] != []]
            value_cache = [current.value_cache[idx] for current in splits if current.value_cache[idx] != []]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]
            
            
# class StaticCache(Cache):
#     """
#     Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

#     Parameters:
#         config (`PretrainedConfig`):
#             The configuration file defining the shape-related attributes required to initialize the static cache.
#         max_batch_size (`int`):
#             The maximum batch size with which the model will be used. Note that a new instance must be instantiated if a
#             smaller batch size is used. If you are manually setting the batch size, make sure to take into account the
#             number of beams if you are running beam search
#         max_cache_len (`int`, *optional*):
#             The maximum sequence length with which the model will be used.
#         device (`torch.device` or `str`, *optional*):
#             The device on which the cache should be initialized. If you're using more than 1 computation device, you
#             should pass the `layer_device_map` argument instead.
#         dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
#             The default `dtype` to use when initializing the layer.
#         layer_device_map (`Optional[Dict[int, Union[str, torch.device, int]]]]`, *optional*):
#             Mapping between the layers and its device. This is required when you are manually initializing the cache
#             and the model is split between different gpus. You can know which layers mapped to which device by
#             checking the associated device_map: `model.hf_device_map`.


#     Example:

#         ```python
#         >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

#         >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
#         >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

#         >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

#         >>> # Prepare a cache class and pass it to model's forward
#         >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
#         >>> max_generated_length = inputs.input_ids.shape[1] + 10
#         >>> past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
#         >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
#         >>> outputs.past_key_values # access cache filled with key/values from generation
#         StaticCache()
#         ```
#     """

#     is_compileable = True

#     def __init__(
#         self,
#         config: PretrainedConfig,
#         max_batch_size: int,
#         max_cache_len: Optional[int] = None,
#         device: Union[torch.device, str, None] = None,
#         dtype: torch.dtype = torch.float32,
#         layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
#     ) -> None:
#         super().__init__()
#         self.max_batch_size = max_batch_size
#         self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len

#         # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
#         self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

#         self._dtype = dtype
#         self.num_key_value_heads = (
#             config.num_attention_heads
#             if getattr(config, "num_key_value_heads", None) is None
#             else config.num_key_value_heads
#         )

#         self.key_cache: List[torch.Tensor] = []
#         self.value_cache: List[torch.Tensor] = []
#         # Note: There will be significant perf decrease if switching to use 5D tensors instead.
#         cache_shape = (self.max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
#         device = torch.device(device) if device is not None else None
#         for idx in range(config.num_hidden_layers):
#             if layer_device_map is not None:
#                 layer_device = layer_device_map[idx]
#             else:
#                 layer_device = device
#             new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
#             new_layer_value_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
#             # Note: `mark_static_address` is used to tag the cache as a fixed data pointer,
#             # preventing compiled graph breaks when updating the cache.
#             torch._dynamo.mark_static_address(new_layer_key_cache)
#             torch._dynamo.mark_static_address(new_layer_value_cache)
#             self.key_cache.append(new_layer_key_cache)
#             self.value_cache.append(new_layer_value_cache)

#     def update(
#         self,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         layer_idx: int,
#         cache_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
#         It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

#         Parameters:
#             key_states (`torch.Tensor`):
#                 The new key states to cache.
#             value_states (`torch.Tensor`):
#                 The new value states to cache.
#             layer_idx (`int`):
#                 The index of the layer to cache the states for.
#             cache_kwargs (`Dict[str, Any]`, `optional`):
#                 Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
#                 to know how where to write in the cache.

#         Return:
#             A tuple containing the updated key and value states.
#         """
#         if cache_kwargs is None:
#             cache_kwargs = {}

#         key_states = key_states.to(self.key_cache[layer_idx].dtype)
#         value_states = value_states.to(self.value_cache[layer_idx].dtype)
#         return _static_cache_update(
#             self.key_cache[layer_idx],
#             self.value_cache[layer_idx],
#             key_states,
#             value_states,
#             cache_kwargs.get("cache_position"),
#         )

#     def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
#         """Returns the sequence length of the cached states that were seen by the model."""
#         # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
#         # limit the check to the first batch member and head dimension.
#         # TODO: deprecate this function in favor of `cache_position`
#         return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

#     def get_max_cache_shape(self) -> Optional[int]:
#         return self.max_cache_len

#     def reset(self):
#         """Resets the cache values while preserving the objects"""
#         for layer_idx in range(len(self.key_cache)):
#             # In-place ops prevent breaking the static address
#             self.key_cache[layer_idx].zero_()
#             self.value_cache[layer_idx].zero_()
            
#     def init_from_dynamic_caches(self, dynamic_caches: List[DynamicCache]):
#         """
#         Updates the static cache with data from multiple dynamic caches for a batch of samples.
        
#         Args:
#             dynamic_caches (List[DynamicCache]): List of dynamic caches for each sample
#             cache_positions (Optional[List[torch.LongTensor]]): List of position indices for each sample
#         """
#         if len(dynamic_caches) > self.max_batch_size:
#             raise ValueError(f"Number of dynamic caches ({len(dynamic_caches)}) exceeds max_batch_size ({self.max_batch_size})")
        
#         for sample_idx, dynamic_cache in enumerate(dynamic_caches):
#             for layer_idx, (key_states, value_states) in enumerate(zip(dynamic_cache.key_cache, dynamic_cache.value_cache)):
#                 self.key_cache[layer_idx][sample_idx][:, :key_states.shape[-2], :].copy_(key_states)
#                 self.value_cache[layer_idx][sample_idx][:, :value_states.shape[-2], :].copy_(value_states)


class RecursiveDynamicCache(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    def __init__(self, base_depth: int, num_recursion: int, sharing: str, update_cache: bool = False) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        self.base_depth = base_depth
        self.num_recursion = num_recursion
        self.sharing = sharing
        self.update_cache = update_cache
        assert sharing in ["cycle", "middle_cycle", "sequence"], f"Invalid sharing type: {sharing}. Must be one of ['cycle', 'middle_cycle', 'sequence']"

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        
        if self.sharing == "cycle":
            return self.key_cache[layer_idx % self.base_depth], self.value_cache[layer_idx % self.base_depth]
        elif self.sharing == "middle_cycle":
            if layer_idx == self.base_depth * self.num_recursion + 1:
                return self.key_cache[self.base_depth + 1], self.value_cache[self.base_depth + 1]
            else:
                return self.key_cache[1 + (layer_idx - 1) % self.base_depth], self.value_cache[1 + (layer_idx - 1) % self.base_depth]
        elif self.sharing == "sequence":
            return self.key_cache[layer_idx // self.base_depth], self.value_cache[layer_idx // self.base_depth]

    def _update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif (
            len(self.key_cache[layer_idx]) == 0
        ):  # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if self.sharing == "cycle":
                if layer_idx < self.base_depth:
                    self._update(key_states, value_states, layer_idx)
                else:
                    return self.key_cache[layer_idx % self.base_depth], self.value_cache[layer_idx % self.base_depth]
                
            elif self.sharing == "middle_cycle":
                if layer_idx < self.base_depth + 1 or layer_idx == self.base_depth * self.num_recursion + 1:
                    if layer_idx == self.base_depth * self.num_recursion + 1:
                        layer_idx = self.base_depth + 1
                    self._update(key_states, value_states, layer_idx)
                else:
                    if not self.update_cache:
                        return self.key_cache[1 + (layer_idx - 1) % self.base_depth], self.value_cache[1 + (layer_idx - 1) % self.base_depth]
                    else:
                        assert "selected_tokens" in cache_kwargs, "selected_tokens must be provided when update_cache is True"
                        _, num_heads, _, head_dim = key_states.shape
                        selected_tokens = cache_kwargs.get("selected_tokens")
                        selected_tokens = selected_tokens.unsqueeze(1).expand(-1, num_heads, -1, head_dim)
                        
                        key_cache = torch.scatter(
                            self.key_cache[1 + (layer_idx - 1) % self.base_depth],
                            dim=2,
                            index=selected_tokens,
                            src=torch.gather(key_states, dim=2, index=selected_tokens),
                        )
                        value_cache = torch.scatter(
                            self.value_cache[1 + (layer_idx - 1) % self.base_depth],
                            dim=2,
                            index=selected_tokens,
                            src=torch.gather(value_states, dim=2, index=selected_tokens),
                        )
                        return key_cache, value_cache
                        
            elif self.sharing == "sequence":
                if layer_idx % self.num_recursion == 0:
                    layer_idx = layer_idx // self.num_recursion
                    self._update(key_states, value_states, layer_idx)
                else:
                    return self.key_cache[layer_idx // self.num_recursion], self.value_cache[layer_idx // self.num_recursion]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]