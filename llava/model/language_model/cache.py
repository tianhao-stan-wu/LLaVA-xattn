from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        for _ in range(config.num_hidden_layers):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            try:
                # If using several devices (e.g.: multiple GPUs), we need to ensure everything is on the same one
                cache_position.to(device=k_out.device)
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # The operator 'aten::index_copy.out' is not currently implemented for the MPS device.
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

        return k_out, v_out