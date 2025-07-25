import torch
from torch import Tensor


@torch.jit.script
def validate_real(tensor: Tensor, name: str) -> None:
    if tensor.is_complex():
        raise ValueError(f"Expected {name} to be real, got dtype {tensor.dtype}")


@torch.jit.script
def validate_at_least_nd(tensor: Tensor, name: str, min_dims: int) -> None:
    if tensor.ndim < min_dims:
        raise ValueError(
            "Expected at least "
            f"{min_dims} dimensions for {name}, got shape {tensor.shape}"
        )


@torch.jit.script
def validate_4d(tensor: Tensor, name: str) -> None:  # for rope frequences
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4 dimensions for {name}, got shape {tensor.shape}")


@torch.jit.script
def validate_same_ndims(
    tensor_1: Tensor, name_1: str, tensor_2: Tensor, name_2: str
) -> None:
    if tensor_1.ndim != tensor_2.ndim:
        raise ValueError(
            "Expected " + name_1 + " and " + name_2 + " to have the same number of "
            f"dims, got shapes {tensor_1.shape} and {tensor_2.shape}",
        )


@torch.jit.script
def validate_head_dim(embeddings: Tensor, rope_encoding: Tensor) -> None:
    if embeddings.size(-1) != rope_encoding.size(-1) * 2:
        raise ValueError(
            "Expected rope_encoding to have last dimension equal to 1/2 embedding's "
            f"head dim, got {rope_encoding.size(-1)} and {embeddings.size(-1)}"
        )


@torch.jit.script
def validate_position_dim(positions: Tensor, rope_freqs: Tensor) -> None:
    if positions.size(-1) != rope_freqs.size(0):
        raise ValueError(
            "Expected first dimension of `rope_freqs` and last dimension of "
            "positions to match, got "
            f"{rope_freqs.size(0)} and {positions.size(-1)}"
        )
