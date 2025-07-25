from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.autograd import Function

from .forward_backward_fns import (
    calculate_rope,
    calculate_rope_backward,
    rotate_embeddings,
    rotate_embeddings_backward,
)


class CalculateRopeFunction(Function):
    """Custom autograd function for memory-efficient RoPE calculation"""

    @staticmethod
    def forward(ctx: Any, positions: Tensor, rope_freqs: Tensor) -> Tensor:
        ctx.save_for_backward(positions, rope_freqs)
        ctx.set_materialize_grads(False)
        return calculate_rope(positions, rope_freqs)

    @staticmethod
    def backward(
        ctx: Any, grad_rope_encoding: Tensor
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        positions, rope_freqs = ctx.saved_tensors

        needs_grad_positions = ctx.needs_input_grad[0]
        needs_grad_rope_freqs = ctx.needs_input_grad[1]

        if grad_rope_encoding is None:
            return None, None

        return calculate_rope_backward(
            grad_rope_encoding,
            positions,
            rope_freqs,
            needs_grad_positions,
            needs_grad_rope_freqs,
        )


class RotateEmbeddingsFunction(Function):
    """Custom autograd function for memory-efficient embedding rotation"""

    @staticmethod
    def forward(ctx: Any, embeddings: Tensor, rope_encoding: Tensor) -> Tensor:
        ctx.save_for_backward(embeddings, rope_encoding)
        ctx.set_materialize_grads(False)
        return rotate_embeddings(embeddings, rope_encoding, needs_autograd=False)

    @staticmethod
    def backward(
        ctx: Any, grad_embeddings_rotated: Tensor
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        embeddings, rope_encoding = ctx.saved_tensors

        needs_grad_embeddings = ctx.needs_input_grad[0]
        needs_grad_rope_encoding = ctx.needs_input_grad[1]

        if grad_embeddings_rotated is None:
            return None, None

        return rotate_embeddings_backward(
            grad_embeddings_rotated,
            embeddings,
            rope_encoding,
            needs_grad_embeddings,
            needs_grad_rope_encoding,
            needs_autograd=False,
        )


class ApplyRoPEFunction(Function):
    """Custom autograd function for memory-efficient end-to-end application of
    RoPE from embeddings, positions, and frequencies.
    """

    @staticmethod
    def forward(
        ctx: Any, embeddings: Tensor, positions: Tensor, rope_freqs: Tensor
    ) -> Tensor:
        ctx.save_for_backard(embeddings, positions, rope_freqs)
        ctx.set_materialize_grads(False)

        rope_encoding = calculate_rope(positions, rope_freqs)
        embeddings_rotated = rotate_embeddings(
            embeddings,
            rope_encoding,
            needs_autograd=False,
        )
        return embeddings_rotated

    @staticmethod
    def backward(
        ctx: Any, grad_embeddings_rotated: Tensor
    ) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        embeddings, positions, rope_freqs = ctx.saved_tensors

        if grad_embeddings_rotated is None:
            return None, None, None

        rope_encoding = calculate_rope(positions, rope_freqs)

        needs_grad_embeddings = ctx.needs_input_grad[0]
        needs_grad_positions = ctx.needs_input_grad[1]
        needs_grad_rope_freqs = ctx.needs_input_grad[2]

        grad_embeddings, grad_rope_encoding = rotate_embeddings_backward(
            grad_embeddings_rotated,
            embeddings,
            rope_encoding,
            needs_grad_embeddings,
            needs_grad_rope_encoding=needs_grad_positions or needs_grad_rope_freqs,
            needs_autograd=False,
        )

        if grad_rope_encoding is None:
            assert not needs_grad_positions and not needs_grad_rope_freqs
            return grad_embeddings, None, None

        grad_positions, grad_rope_freqs = calculate_rope_backward(
            grad_rope_encoding,
            positions,
            rope_freqs,
            needs_grad_positions,
            needs_grad_rope_freqs,
        )

        return grad_embeddings, grad_positions, grad_rope_freqs


def calculate_rope_checkpointed(positions: Tensor, rope_freqs: Tensor) -> Tensor:
    """Memory-efficient differentiable version of ``calculate_rope``.

    This wrapper avoids building a full autograd graph inside the scripted
    kernel and stores only the ``positions`` and ``rope_freqs`` tensors required
    by the backward formula. This results in potentially large memory savings
    if the sequence length is large.
    """
    out = CalculateRopeFunction.apply(positions, rope_freqs)
    return out  # pyright: ignore[reportReturnType]


def rotate_embeddings_checkpointed(embeddings: Tensor, rope_encoding: Tensor) -> Tensor:
    """Memory-efficient differentiable version of ``rotate_embeddings``.

    The forward path calls the fast kernel with ``needs_autograd=False`` so it
    can apply its in-place optimizations; gradients are supplied by a bespoke
    backward kernel.  In practice this saves one extra copy of the
    ``embeddings`` tensor compared to the naïve autograd graph, which again can
    save significant memory if the sequence length is large.
    """
    out = RotateEmbeddingsFunction.apply(embeddings, rope_encoding)
    return out  # pyright: ignore[reportReturnType]


def apply_rope_checkpointed(
    embeddings: Tensor, positions: Tensor, rope_freqs: Tensor
) -> Tensor:
    """End-to-end rotary positional encoding (RoPE) in a single autograd node.

    Internally, this function computes the full RoPE encoding tensor and applies it
    to the embeddings, without storing it for backprop. Since this tensor is potentially
    very large for large sequence length and/or embedding dimension but is cheap to
    calculate with just one broadcasted multiplication, this gradient checkpointing
    logic can trade off significant memory savings for a small computation increase.

    Args:
        embeddings (Tensor): Embeddings tensor to be rotated (usually a query or
            key tensor) of real dtype and shape [..., n_heads, head_dim]
        positions (Tensor): Position information for each embedding element of shape
            [..., position_dim], where ... are arbitrary batch dimensions and
            position_dim is the dimensionality of the position representation.
        rope_freqs (Tensor): Frequency values for rotary encodings of shape
            [position_dim, n_freq_groups, n_heads, head_dim/2], where n_freq_groups
            and n_heads can be 1 for broadcasting.

    Returns:
        embeddings_rotated (Tensor): Embedding tensor after rotation, of shape
            [..., n_heads, head_dim] and same dtype as `embeddings`.
    """
    out = ApplyRoPEFunction.apply(embeddings, positions, rope_freqs)
    return out  # pyright: ignore[reportReturnType]
