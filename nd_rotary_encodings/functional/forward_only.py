from typing import Optional, Union

import torch
from torch import Tensor

from nd_rotary_encodings.functional.forward_backward_fns import (
    calculate_rope,
    rotate_embeddings,
)


@torch.no_grad()
def rotate_embeddings_forward_only(
    embeddings: Tensor, rope_encoding: Tensor, inplace: bool = False
) -> Tensor:
    """Forward-only version of ``rotate_embeddings``.

    This calls `rotate_embeddings` with additional optimizations like in-place tensor
    operations that make it incompatible with autograd.

    Args:
        embeddings (Tensor): Embeddings tensor to be rotated (usually a query or
            key tensor) of real dtype and shape [..., n_heads, head_dim]
        rope_encoding (Tensor): Position encoding of real dtype and shape
            [..., n_heads, head_dim/2] or
            [..., 1,       head_dim/2] (broadcasted over heads)
        inplace: (bool): If True, the supplied `embeddings` tensor is rotated in-place
            (i.e., overwritten with the new values) for maximum memory efficiency.
            Default: False.

    Returns:
        embeddings_rotated (Tensor): Embedding tensor after rotation, of shape
            [..., n_heads, head_dim] and real dtype
    """
    if not inplace:
        embeddings = embeddings.clone()
    return rotate_embeddings(embeddings, rope_encoding, needs_autograd=False)


@torch.no_grad()
def apply_rope_forward_only(
    embeddings: Tensor,
    positions: Tensor,
    rope_freqs: Tensor,
    inplace: bool = False,
    self_attn_key_embeddings: Optional[Tensor] = None,
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """End-to-end rotary positional encoding (RoPE) in a single autograd node.

    This calls `calculate_rope` and `rotate_embeddings` with additional optimizations
    like in-place tensor operations that make it incompatible with autograd.

    Args:
        embeddings (Tensor): Embeddings tensor to be rotated (usually a query or
            key tensor) of real dtype and shape [..., n_heads, head_dim]
        positions (Tensor): Position information for each embedding element of shape
            [..., position_dim], where ... are arbitrary batch dimensions and
            position_dim is the dimensionality of the position representation.
        rope_freqs (Tensor): Frequency values for rotary encodings of shape
            [position_dim, n_freq_groups, n_heads, head_dim/2], where n_freq_groups
            and n_heads can be 1 for broadcasting.
        inplace: (bool): If True, the supplied `embeddings` tensor is rotated in-place
            (i.e., overwritten with the new values) for maximum memory efficiency.
            Default: False.

    Returns:
        embeddings_rotated (Tensor): Embedding tensor after rotation, of shape
            [..., n_heads, head_dim] and same dtype as `embeddings`.
    """
    rope_encoding = calculate_rope(positions, rope_freqs)
    embeddings_rotated = rotate_embeddings_forward_only(
        embeddings, rope_encoding, inplace=inplace
    )

    if self_attn_key_embeddings is None:
        return embeddings_rotated

    key_embeddings_rotated = rotate_embeddings_forward_only(
        self_attn_key_embeddings, rope_encoding, inplace=inplace
    )
    return embeddings_rotated, key_embeddings_rotated
