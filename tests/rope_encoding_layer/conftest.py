from typing import Any, Union, Optional

import pytest
import torch
from torch import Tensor
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, array_shapes

from nd_rotary_encodings import (
    RoPEEncodingND,
)


@st.composite
def positions_strategy(draw):
    positions_dtype = draw(st.sampled_from([torch.float32, torch.long]))
    if positions_dtype == torch.float32:
        min_position = draw(
            st.floats(min_value=-1e30, max_value=1e30, exclude_max=True)
        )
        max_position = draw(st.floats(min_value=min_position, max_value=1e30))
    else:
        min_position = draw(st.integers(min_value=int(-1e10), max_value=int(1e10) - 1))
        max_position = draw(st.integers(min_value=min_position, max_value=int(1e10)))
    return positions_dtype, min_position, max_position


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Base configuration for RoPEEncodingND tests."""
    return {
        "position_dim": 2,
        "embed_dim": 16,
        "n_heads": 4,
        "dtype": torch.float32,
    }


@pytest.fixture
def sample_data(base_config: dict[str, Any], device: str):
    """Sample input data for testing forward passes."""
    batch_size = 2
    seq_len = 10

    return {
        "query": torch.randn(
            batch_size,
            seq_len,
            base_config["embed_dim"],
            dtype=base_config["dtype"],
            device=device,
        ),
        "query_pos": torch.rand(
            batch_size,
            seq_len,
            base_config["position_dim"],
            dtype=base_config["dtype"],
            device=device,
        )
        * 10,  # Unnormalized positions
        "key": torch.randn(
            batch_size,
            seq_len,
            base_config["embed_dim"],
            dtype=base_config["dtype"],
            device=device,
        ),
        "key_pos": torch.rand(
            batch_size,
            seq_len,
            base_config["position_dim"],
            dtype=base_config["dtype"],
            device=device,
        )
        * 10,  # Unnormalized positions
    }


@st.composite
def rope_config_strategy(draw: st.DrawFn, require_grads: bool = False):
    position_dim = draw(st.integers(1, 4))
    n_heads = draw(st.integers(1, 8))
    embed_dim = draw(st.integers(1, 8)) * 2 * n_heads
    share_heads = draw(st.booleans())
    n_freq_groups = draw(st.integers(1, 4))
    enforce_freq_groups_equal = (embed_dim // n_heads) % (n_freq_groups * 2) == 0

    def will_rotate(freq_group_pattern: np.ndarray):
        if not enforce_freq_groups_equal:
            return freq_group_pattern[0].any().item()
        return freq_group_pattern.any().item()

    freq_group_pattern = draw(
        arrays(
            np.bool,
            shape=(n_freq_groups, position_dim),
            elements=st.booleans(),
            fill=None,
        ).filter(will_rotate)
    )
    rope_base_theta = draw(
        st.one_of(
            st.floats(min_value=1.0, max_value=1e10, exclude_min=True),
            arrays(
                np.float32,
                shape=(
                    draw(st.sampled_from([1, n_freq_groups])),
                    draw(st.sampled_from([1, position_dim])),
                ),
                elements=st.floats(min_value=1.0, max_value=1e10, exclude_min=True),
                fill=None,
            ),
        )
    )
    dtype = draw(st.just(torch.float32))

    # Optimizations config
    use_checkpointing = forward_only = inplace = False
    mode = st.sampled_from(["normal", "checkpointing", "forward_only", "inplace"])
    if mode == "checkpointing":
        use_checkpointing = True
    elif mode == "forward_only":
        forward_only = True
    elif mode == "inplace":
        forward_only = True
        inplace = True

    # Tensor creation config
    # Embedding value range
    embedding_min_value = draw(st.floats(-1e15, 1e15, exclude_max=True))
    embedding_max_value = draw(st.floats(embedding_min_value, 1e15))

    # shared helper function for position grid shape
    def _position_grid_shape() -> tuple[int, ...]:
        return draw(
            array_shapes(
                min_dims=position_dim, max_dims=position_dim, min_side=2, max_side=8
            )
        )

    # Query positions: Random or use grid
    use_position_grid_query = draw(st.booleans())
    if use_position_grid_query:
        position_dtype_query = draw(st.none())
        position_min_query = draw(st.none())
        position_max_query = draw(st.none())
        # Need to have at least position_dim dims before embed_dim when using
        # position grid
        position_shape_query = _position_grid_shape()
        batch_shape_query = draw(array_shapes(min_side=0)) + position_shape_query
    else:
        position_dtype_query, position_min_query, position_max_query = draw(
            positions_strategy().filter(lambda drawn: drawn[1] <= 0 or drawn[2] >= 1)
        )
        batch_shape_query = draw(array_shapes(min_side=0))

    # Decide whether to include separate key embeddings
    include_key = draw(st.booleans())
    if include_key:
        # Use separate position tensor for key or share query positions
        include_key_pos = draw(st.booleans())
        if include_key_pos:
            # For generality, allow key tensor to have different shape from query
            # Key positions: random or use grid
            use_position_grid_key = draw(st.booleans())
            if use_position_grid_key:
                position_dtype_key = draw(st.none())
                position_min_key = draw(st.none())
                position_max_key = draw(st.none())
                # Key tensor needs position_dim dims before embed_dim, like query
                position_shape_key = _position_grid_shape()
                batch_shape_key = draw(array_shapes(min_side=0)) + position_shape_key
            else:  # non-grid positions
                position_dtype_key, position_min_key, position_max_key = draw(
                    positions_strategy().filter(
                        lambda drawn: drawn[1] <= 0 or drawn[2] >= 1
                    )
                )
                batch_shape_key = draw(array_shapes(min_side=0))
        else:  # key_pos=None (reuse query positions)
            batch_shape_key = draw(st.just(batch_shape_query))
            use_position_grid_key = draw(st.just(False))
            position_dtype_key = draw(st.none())
            position_min_key = draw(st.none())
            position_max_key = draw(st.none())
    else:
        include_key_pos = draw(st.just(False))
        use_position_grid_key = draw(st.just(False))
        batch_shape_key = draw(st.none())
        position_dtype_key = draw(st.none())
        position_min_key = draw(st.none())
        position_max_key = draw(st.none())

    seed = draw(st.integers(min_value=0, max_value=int(1e8)))

    def is_float(dtype: Union[torch.dtype, None]):
        return dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]

    # Decide if tensors should require grads
    query_require_grad = query_pos_require_grad = False
    key_require_grad = key_pos_require_grad = False
    if require_grads:
        query_require_grad = draw(st.booleans())
        if is_float(position_dtype_query):
            query_pos_require_grad = draw(st.booleans())
        if include_key:
            key_require_grad = draw(st.booleans())
        if is_float(position_dtype_key):
            key_pos_require_grad = draw(st.booleans())

    return {
        "config": {
            "position_dim": position_dim,
            "embed_dim": embed_dim,
            "n_heads": n_heads,
            "share_heads": share_heads,
            "freq_group_pattern": freq_group_pattern,
            "enforce_freq_groups_equal": enforce_freq_groups_equal,
            "rope_base_theta": rope_base_theta,
            "use_checkpointing": use_checkpointing,
            "forward_only": forward_only,
            "inplace": inplace,
            "dtype": dtype,
        },
        "tensor_config": {
            "embed_dim": embed_dim,
            "embedding_min_value": embedding_min_value,
            "embedding_max_value": embedding_max_value,
            "position_dim": position_dim,
            "batch_shape_query": batch_shape_query,
            "use_position_grid_query": use_position_grid_query,
            "position_dtype_query": position_dtype_query,
            "position_min_query": position_min_query,
            "position_max_query": position_max_query,
            "include_key": include_key,
            "batch_shape_key": batch_shape_key,
            "include_key_pos": include_key_pos,
            "use_position_grid_key": use_position_grid_key,
            "position_dtype_key": position_dtype_key,
            "position_min_key": position_min_key,
            "position_max_key": position_max_key,
            "seed": seed,
            "query_require_grad": query_require_grad,
            "query_pos_require_grad": query_pos_require_grad,
            "key_require_grad": key_require_grad,
            "key_pos_require_grad": key_pos_require_grad,
        },
    }


def rope_input_tensors(
    embed_dim: int,
    embedding_min_value: float,
    embedding_max_value: float,
    position_dim: int,
    batch_shape_query: tuple[int, ...],
    use_position_grid_query: bool,
    position_dtype_query: torch.dtype,
    position_min_query: Union[int, float],
    position_max_query: Union[int, float],
    include_key: bool,
    batch_shape_key: Optional[tuple[int, ...]],
    include_key_pos: bool,
    use_position_grid_key: Optional[bool],
    position_dtype_key: Optional[torch.dtype],
    position_min_key: Optional[Union[int, float]],
    position_max_key: Optional[Union[int, float]],
    seed: int,
    query_require_grad: bool,
    query_pos_require_grad: bool,
    key_require_grad: bool,
    key_pos_require_grad: bool,
    device: Union[str, torch.device],
    force_unnormalized_positions: bool = True,
):
    device = torch.device(device)

    # save rng state and set seed
    if device.type == "cuda":
        rng_state = torch.cuda.get_rng_state(device)
    else:
        rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    # Helper function for creating non-grid position tensor
    def _random_positions(
        batch_shape: tuple[int, ...],
        dtype: torch.dtype,
        pos_min: Union[int, float],
        pos_max: Union[int, float],
        require_grad: bool,
    ):
        pos_tensor = torch.empty(
            batch_shape + (position_dim,),
            device=device,
            dtype=dtype,
        )
        if torch.is_floating_point(pos_tensor):
            pos_tensor.uniform_(pos_min, pos_max)
        else:
            pos_tensor.random_(int(pos_min), int(pos_max) + 1)
        if force_unnormalized_positions and pos_tensor.numel() > 0:
            while pos_tensor.min() > 0.0 and pos_tensor.max() <= 1.0:
                pos_tensor = pos_tensor + 1
        if require_grad:
            pos_tensor.requires_grad_(True)
        return pos_tensor

    # Create query embeddings and position tensor
    query = torch.empty(batch_shape_query + (embed_dim,), device=device).uniform_(
        embedding_min_value, embedding_max_value
    )
    query.requires_grad_(query_require_grad)
    if use_position_grid_query:
        query_pos = RoPEEncodingND.position_grid(
            query.shape, query.ndim - position_dim - 1, device=device
        )
        assert query_pos.size(-1) == position_dim
    else:
        query_pos = _random_positions(
            batch_shape_query,
            position_dtype_query,
            position_min_query,
            position_max_query,
            query_pos_require_grad,
        )

    # Create key embeddings and position tensor if needed
    if include_key:
        assert batch_shape_key is not None
        key = torch.empty(batch_shape_key + (embed_dim,), device=device).uniform_(
            embedding_min_value,
            embedding_max_value,
        )
        key.requires_grad_(key_require_grad)
        if include_key_pos:
            if use_position_grid_key:
                key_pos = RoPEEncodingND.position_grid(
                    key.shape, key.ndim - position_dim - 1, device=device
                )
                assert key_pos.size(-1) == position_dim
            else:
                assert position_dtype_key is not None
                assert position_min_key is not None
                assert position_max_key is not None
                key_pos = _random_positions(
                    batch_shape_key,
                    position_dtype_key,
                    position_min_key,
                    position_max_key,
                    key_pos_require_grad,
                )
        else:
            key_pos = None
    else:
        key = None
        key_pos = None

    # reset rng state
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng_state, device)
    else:
        torch.set_rng_state(rng_state)

    return {
        "query": query,
        "query_pos": query_pos,
        "key": key,
        "key_pos": key_pos,
    }


def allclose_zero(tensor: Tensor):
    return torch.allclose(tensor, torch.zeros_like(tensor))
