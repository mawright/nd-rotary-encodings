import warnings
from typing import Optional, Union, TYPE_CHECKING

import torch
from torch import Tensor, nn

from ..functional.forward_backward_fns import (
    calculate_rope,
    rotate_embeddings,
)
from ..functional.autograd import apply_rope_checkpointed
from .freq_init import init_nd_freqs
from .utils import can_broadcast_shapes

# Based on code from
# https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py


class RoPEEncodingND(nn.Module):
    """N-dimensional Rotary Position Embedding (RoPE) module.

    Implements rotary position embeddings for arbitrary dimensional positional inputs.
    This module applies RoPE to queries and keys in attention mechanisms, enabling
    position-aware attention across N spatial dimensions.

    Args:
        position_dim (int): Number of position dimensions (e.g., 2 for 2D, 3 for 3D).
        embed_dim (int): Total embedding dimension, must be divisible by n_heads.
        n_heads (int): Number of attention heads.
        share_heads (bool): If True, then only one set of frequencies per frequency
            group is created, that is shared among all attention heads, similar to
            traditional 1D RoPE. Defaults to False.
        freq_group_pattern (Optional[Tensor]): Boolean tensor of shape
            [n_freq_groups, position_dim] defining frequency group inclusion. The
            (head_dim/2) elements of the RoPE encoding vector will be split among the
            frequency groups. If position i, j is True, then frequency group i
            includes position dimension j. If None, freq_group_pattern will default
            to an all-True tensor of shape [1, position_dim]; i.e., one frequency group
            with all position dimensions.
        enforce_freq_groups_equal (boolean): If True, then this function will raise
            a ValueError if the (head_dim/2) available elements of the RoPE vector
            cannot be evenly split between the frequency groups. If False, then
            trailing frequency groups may have fewer RoPE encodings assigned to them.
        rope_base_theta (Union[Tensor], float]): Base value(s) for frequency scaling.
            Can be a single float (applied to all dimensions and frequency groups)
            or a 2D tensor of shape [n_freq_groups, position_dim], with either
            component dimension allowed to be 1 for broadcasting. Entries corresponding
            to non-included position dimensions in a frequency group will be ignored.
            Larger values of theta result in lower-frequency rotations, and may be more
            suitable for dimensions of greater spatial scale. Default: 10.0
        dtype (torch.dtype): Data type for the internal parameters. Default: torch.float
    """

    if TYPE_CHECKING:
        freq_group_pattern: Tensor
        freq_pos_indices: Tensor
        freq_group_indices: Tensor
        freq_head_indices: Tensor
        freq_enc_indices: Tensor
        encoding_ranges: Tensor

    def __init__(
        self,
        position_dim: int,
        embed_dim: int,
        n_heads: int,
        share_heads: bool = False,
        freq_group_pattern: Optional[Tensor] = None,
        enforce_freq_groups_equal: bool = True,
        rope_base_theta: Union[Tensor, float, list[list[float]]] = 10.0,
        use_checkpointing: bool = False,
        dtype=torch.float,
    ):
        """Initialize the module"""
        super().__init__()
        self.embed_dim = embed_dim
        if embed_dim % n_heads != 0:
            raise ValueError(
                "Expected embed_dim to be divisible by n_heads, got "
                f"{embed_dim} and {n_heads}"
            )
        self.head_dim = embed_dim // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"Expected head_dim to be divisible by 2, got {self.head_dim}"
            )
        self.position_dim = position_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.share_heads = share_heads
        self.use_checkpointing = use_checkpointing

        if freq_group_pattern is None:
            # default frequency group pattern: one group with all position dimensions
            freq_group_pattern = torch.ones(1, position_dim, dtype=torch.bool)
        freq_group_pattern = torch.as_tensor(freq_group_pattern, dtype=torch.bool)

        self.enforce_freq_groups_equal = enforce_freq_groups_equal

        self.validate_freq_group_pattern(freq_group_pattern)
        self.register_buffer("freq_group_pattern", freq_group_pattern)
        self.n_freq_groups = freq_group_pattern.size(0)

        self._base_theta = torch.as_tensor(rope_base_theta, dtype=dtype)
        self.dtype = dtype
        self._init_freq_param()

    def _init_freq_param(self):
        """Initialize the frequency parameters for the RoPE module.

        Creates and stores the frequency parameters as trainable parameters and
        precomputes the indices used to construct the full sparse RoPE frequency
        tensor.
        """
        effective_n_heads = self.n_heads if not self.share_heads else 1

        freqs, encoding_ranges = init_nd_freqs(
            self.position_dim,
            self.head_dim,
            effective_n_heads,
            self.freq_group_pattern,
            self.enforce_freq_groups_equal,
            self._base_theta,
            dtype=self.dtype,
        )
        self.validate_grouped_freqs(freqs, encoding_ranges)

        self.freqs = nn.ParameterList(freqs)
        self.register_buffer("encoding_ranges", encoding_ranges)

        # Precompute indices for grouped_rope_freqs_tensor
        indices_list = []

        for g, range in enumerate(encoding_ranges):
            range_start, range_end = (int(r) for r in range)
            range_size = range_end - range_start
            pos_dims = torch.nonzero(self.freq_group_pattern[g], as_tuple=True)[0]

            # Create indexing tensors for this frequency group
            # Order matches output tensor shape: [position_dim, n_freq_groups, n_heads, head_dim//2]
            pos_idx = pos_dims.view(-1, 1, 1).expand(-1, effective_n_heads, range_size)
            g_idx = torch.full(
                (pos_dims.size(0), effective_n_heads, range_size),
                g,
                dtype=torch.long,
                device=pos_dims.device,
            )
            head_idx = (
                torch.arange(effective_n_heads, device=pos_dims.device)
                .view(1, -1, 1)
                .expand(pos_dims.size(0), -1, range_size)
            )
            dim_idx = (
                torch.arange(range_start, range_end, device=pos_dims.device)
                .view(1, 1, -1)
                .expand(pos_dims.size(0), effective_n_heads, -1)
            )

            # Stack with dimension order matching output tensor
            indices = torch.stack(
                [
                    pos_idx.flatten(),
                    g_idx.flatten(),
                    head_idx.flatten(),
                    dim_idx.flatten(),
                ],
                dim=0,
            )
            indices_list.append(indices)

        # Concatenate all indices
        indices = torch.cat(indices_list, dim=1)

        # store indices for construction ofm freq tensor in forward pass
        pos_indices, group_indices, head_indices, enc_indices = indices.unbind(0)
        self.register_buffer("freq_pos_indices", pos_indices)
        self.register_buffer("freq_group_indices", group_indices)
        self.register_buffer("freq_head_indices", head_indices)
        self.register_buffer("freq_enc_indices", enc_indices)

    def validate_freq_group_pattern(self, freq_group_pattern: Tensor):
        if freq_group_pattern.ndim != 2:
            raise ValueError(
                "Expected 2D tensor for freq_group_pattern, got shape "
                f"{freq_group_pattern.size()}"
            )
        if freq_group_pattern.size(1) != self.position_dim:
            raise ValueError(
                "Expected second dimension of freq_group_pattern to have size equal to "
                f"position_dim, got freq_group_pattern shape {freq_group_pattern.size()} "
                f"and position_dim={self.position_dim}"
            )
        n_freq_groups = freq_group_pattern.size(0)
        half_head_dim = self.head_dim // 2
        remainder = half_head_dim % n_freq_groups

        if remainder > 0 and self.enforce_freq_groups_equal:
            raise ValueError(
                f"RoPE encodings ({half_head_dim}) not evenly divisible by frequency "
                f"groups ({n_freq_groups})"
            )

    def validate_grouped_freqs(self, freqs: list[Tensor], encoding_ranges: Tensor):
        # Validate number of frequency groups
        n_freq_groups = len(freqs)
        if self.freq_group_pattern.size(0) != n_freq_groups:
            raise ValueError(
                "Expected the first dimension of freq_group_pattern (shape: "
                f"{self.freq_group_pattern.shape}) to have size equal to the length of the"
                f"freqs list ({len(freqs)})"
            )

        # Validate head_dim is consistent
        half_head_dim_list = [freqs.size(2) for freqs in freqs]
        if len(set(half_head_dim_list)) != 1 and self.enforce_freq_groups_equal:
            raise ValueError(
                "Expected tensors in freqs to all have the same number of "
                f"RoPE encodings; got {half_head_dim_list}"
            )

        # Validate n_heads is consistent
        n_heads_list = [freqs.size(1) for freqs in freqs]
        n_heads_set = set(n_heads_list)
        if not (
            len(n_heads_set) == 1
            or (len(n_heads_set) == 2 and len(n_heads_set - set((1,))) == 1)
        ):
            raise ValueError(
                "Expected tensors in freqs to have number of attention heads "
                f"all equal and/or 1, got {n_heads_list}"
            )

        # Validate encoding ranges
        if encoding_ranges.size(0) != n_freq_groups:
            raise ValueError(
                "Expected first dim of encoding_ranges to be equal to n_freq_groups "
                f"({n_freq_groups}), got shape {encoding_ranges}"
            )

        if not (
            torch.all(encoding_ranges[:, 0] <= encoding_ranges[:, 1])
            and torch.all(encoding_ranges[:-1, 1] == encoding_ranges[1:, 0])
        ):
            raise ValueError(
                "Expected encoding_ranges to be a 2D tensor of contiguous, "
                f"non-overlapping slices, got {encoding_ranges}"
            )

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Apply rotary position embeddings to query and optionally key tensors.

        Applies position-dependent rotations to query and key tensors based on
        their associated position information.

        Args:
            query (Tensor): Query tensor of shape [..., embed_dim].
            query_pos (Tensor): Position tensor for query of shape
                [..., position_dim]. The leading dimensions must match those of query.
                It is assumed that the positions are NOT normalized to the standard
                [0, 1] range and are instead the true positions.
            key (Optional[Tensor]): Key tensor of shape [..., embed_dim]. Default: None
            key_pos (Optional[Tensor]): Position tensor for key of shape
                [..., position_dim]. If None and key is provided, query_pos will be
                used. It is assumed that the positions are NOT normalized to the
                standard [0, 1] range and are instead the true positions. Default: None

        Returns:
            Union[Tensor, tuple[Tensor, Tensor]]:
                - If key is None: Rotated query tensor of same shape as input query.
                - If key is provided: Tuple of (rotated query, rotated key) tensors.

        Note:
            - For query/key embeddings with a regular grid structure, a default
                position grid may be obtained from the static method `position_grid`.

        Raises:
            ValueError: If the tensor shapes are incompatible.

        Warns:
            UserWarning: If position coordinates appear to be normalized
                (in [0,1] range).
        """

        self.shape_check(query, query_pos)
        if query_pos.numel() > 0 and query_pos.min() > 0.0 and query_pos.max() <= 1.0:
            warnings.warn(
                "Expected un-normalized (i.e., not inside [0,1]) coordinates "
                "for position but found potentially normalized coordinates. "
                "Did you accidentally pass in normalized coordinates?\n(Your coord "
                f"range: [{query_pos.min().item(), query_pos.max().item()}])",
                UserWarning,
            )
        if key_pos is not None:
            assert key is not None
            self.shape_check(key, key_pos)
        freq_tensor = self.grouped_rope_freqs_tensor(self.freqs)

        query_batch_dims = query.shape[:-1]

        # unstack query heads
        query = query.reshape(query_batch_dims + (self.n_heads, self.head_dim))

        if self.use_checkpointing:
            query_rotated = self.apply_rope_checkpointed(query, query_pos, freq_tensor)
        else:
            query_rot_vec = self.calculate_rope(query_pos, freq_tensor)
            query_rotated = self.rotate_embeddings(query, query_rot_vec)

        # stack heads back
        query_rotated = query_rotated.view(query_batch_dims + (self.embed_dim,))

        if key is None:
            return query_rotated

        key_batch_dims = key.shape[:-1]
        # unstack key heads
        key = key.reshape(key_batch_dims + (self.n_heads, self.head_dim))

        if self.use_checkpointing:
            if key_pos is None:
                key_pos = query_pos
            key_rotated = self.apply_rope_checkpointed(key, key_pos, freq_tensor)
        else:
            if key_pos is not None:
                key_rot_vec = self.calculate_rope(key_pos, freq_tensor)
            else:
                key_rot_vec = query_rot_vec
            key_rotated = self.rotate_embeddings(key, key_rot_vec)

        # stack heads back
        key_rotated = key_rotated.view(key_batch_dims + (self.embed_dim,))

        return query_rotated, key_rotated

    @staticmethod
    def position_grid(
        embeddings_shape: Union[tuple[int, ...], Tensor],
        start_dim: int = 1,
        end_dim: int = -1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Generates a regularly-spaced grid of positions based on the input shape.

        This function may be used to generate a tensor of positions corresponding to
        the tensor indices of each element in the embeddings tensor. This is
        potentially useful for regularly-spaced queries and keys, such as embeddings
        corresponding to text tokens or image pixels. The exact form of the position
        grid tensor is torch.stack(torch.meshgrid(
            *[torch.arange(size) for size in embeddings_shape[start_dim:end_dim]],
            indexing="ij"
        ))

        Args:
            embeddings_shape (Tensor): The full shape of the embeddings tensor.
            start_dim (int, optional): Start index of the position dimensions in
                embeddings_shape, inclusive. Defaults to 1 (i.e., one batch dim).
            end_dim (int, optional): End index of the position dimensions in
                embeddings_shape, exclusive. Defaults to -1 (i.e., one feature dim).
            device(str | torch.device, optional): The device on which to create the
                tensor. Defaults to None (i.e., default device).
            dtype(torch.device, optional): The dtype for the created tensor. Defaults
                to None (i.e., default dtype).

        Returns:
            Tensor: Created position grid tensor, of shape
                [*embeddings_shape[start_dim:end_dim],
                 len(embeddings_shape[start_dim:end_dim])]
        """
        grid = torch.stack(
            torch.meshgrid(
                *[
                    torch.arange(int(size), device=device, dtype=dtype)
                    for size in embeddings_shape[start_dim:end_dim]
                ],
                indexing="ij",
            ),
            dim=-1,
        )
        return grid

    def grouped_rope_freqs_tensor(
        self,
        grouped_rope_freqs: Union[list[Tensor], nn.ParameterList],
    ) -> Tensor:
        """Use frequency group information to build the full RoPE frequency tensor that
        is multiplied by the positions to produce RoPE encodings.

        This function takes the per-group RoPE frequencies to construct the RoPE
        frequency tensor. The RoPE frequency tensor has shape
        [position_dim, n_freq_groups, n_heads, head_dim/2], and is zero at positions
        where a position dimension is not included in a frequency group. The
        frequencies are stored separately per frequency group in tensors of shape
        [position_dim_g, n_heads, group_encoding_dim] because each frequency group may
        have a different number of active position dimensions and/or assigned encoding
        dimensions.

        Args:
            grouped_rope_freqs (list[Tensor]): List of per-group frequency tensors, as
                generated by init_nd_freqs, each of shape
                [
                    position_dim_g,
                    n_heads,
                    {head_dim//(2 * n_freq_groups), head_dim//(2 * n_freq_groups) + 1}
                ],
                where position_dim_g is the number of position dimensions included in
                frequency group g.

        Returns:
            Tensor: RoPE frequency tensor of shape
                [position_dim, n_freq_groups, n_heads, head_dim/2] or
                [position_dim, n_freq_groups,       1, head_dim/2], with nonzero
                elements corresponding to position dimensions included in each
                frequency group. It may be passed to `calculate_rope` with the
                positions tensor to compute RoPE encodings.
        """
        if isinstance(grouped_rope_freqs, Tensor):
            grouped_rope_freqs = [grouped_rope_freqs]

        # Create output tensor
        rope_freqs = grouped_rope_freqs[0].new_zeros(
            self.position_dim,
            self.n_freq_groups,
            self.n_heads if not self.share_heads else 1,
            self.head_dim // 2,
        )

        values = torch.cat([fg.flatten() for fg in grouped_rope_freqs])

        rope_freqs.index_put_(
            (
                self.freq_pos_indices,
                self.freq_group_indices,
                self.freq_head_indices,
                self.freq_enc_indices,
            ),
            values,
        )

        return rope_freqs

    @staticmethod
    def calculate_rope(positions: Tensor, rope_freqs: Tensor) -> Tensor:
        """Creates rotation vectors from position coordinates and RoPE frequencies.

        Transforms positional information into rotation vectors for RoPE.

        Args:
            positions (Tensor): Position tensor of shape [..., position_dim].
            rope_freqs (Tensor): Frequency tensor for rotary encodings of shape
                [position_dim, n_freq_groups, n_heads, head_dim/2].

        Returns:
            Tensor: Real-valued positional encodings of shape
                [..., n_heads, head_dim/2].
        """
        return calculate_rope(positions.to(rope_freqs), rope_freqs)

    @staticmethod
    def rotate_embeddings(query_or_key: Tensor, rope_encoding: Tensor) -> Tensor:
        """Applies rotary embeddings to query or key tensor using complex
        multiplication.

        Rotates the query or key tensor using the rotation vectors via complex
        multiplication.

        Args:
            query_or_key (Tensor): Query or key tensor of shape
                [..., n_heads, head_dim].
            rope_encoding (Tensor): Real-valued RoPE encoding tensor of shape
                [..., n_heads, head_dim/2].

        Returns:
            Tensor: Rotated query or key tensor of same shape as input query_or_key.
        """
        # Unsqueeze rope_encoding if needed
        dim_diff = query_or_key.ndim - rope_encoding.ndim
        if dim_diff > 0:
            rope_encoding = rope_encoding.view((1,) * dim_diff + rope_encoding.shape)
        return rotate_embeddings(query_or_key, rope_encoding)

    @staticmethod
    def apply_rope_checkpointed(
        query_or_key: Tensor, positions: Tensor, rope_freqs: Tensor
    ) -> Tensor:
        """Memory-optimized calculation and application of RoPE from components.

        Uses an end-to-end function to calculate RoPE rotation vectors from
        position coordinates and RoPE frequencies, then apply them to query or key
        embeddings.
        Under the hood, this uses a custom autograd Function and explicit backward
        calculation to save memory, at the cost of recalculating the RoPE rotation
        vectors during the backward pass.

        Args:
            query_or_key (Tensor): Query or key tensor of shape
                [..., n_heads, head_dim].
            positions (Tensor): Position tensor of shape [..., position_dim].
            rope_freqs (Tensor): Frequency tensor for rotary encodings of shape
                [position_dim, n_freq_groups, n_heads, head_dim/2].

        Returns:
            Tensor: Rotated query or key tensor of same shape as input query_or_key.
        """
        return apply_rope_checkpointed(query_or_key, positions, rope_freqs)

    def shape_check(self, query_or_key: Tensor, query_or_key_pos: Tensor):
        """Validates the shapes of query/key and their position tensors.

        Args:
            query_or_key (Tensor): Query or key tensor of shape [..., embed_dim].
            query_or_key_pos (Tensor): Position tensor of shape [..., position_dim].
                Must be broadcastable to the shape of query_or_key.

        Raises:
            ValueError: If tensor shapes are incompatible.
        """
        if not can_broadcast_shapes(
            query_or_key.shape[:-1], query_or_key_pos.shape[:-1]
        ):
            raise ValueError(
                "Expected leading dims of query_or_key_pos to be broadcastable to "
                "leading dims of query_or_key, but got shapes "
                f"{query_or_key_pos.shape} and {query_or_key.shape}, respectively."
            )
        if query_or_key.shape[-1] != self.embed_dim:
            raise ValueError(
                "Expected query_or_key to have last dim equal to embed_dim "
                f"(={self.embed_dim}), got {query_or_key.shape[-1]}"
            )
        if query_or_key_pos.shape[-1] != self.position_dim:
            raise ValueError(
                "Expected query_or_key_pos to have last dim equal to pos_dim "
                f"(={self.position_dim}), got {query_or_key_pos.shape[-1]}"
            )

    def reset_parameters(self):
        """Resets frequency parameters"""
        freqs, _ = init_nd_freqs(
            self.position_dim,
            self.head_dim,
            self.n_heads if not self.share_heads else 1,
            self.freq_group_pattern,
            self.enforce_freq_groups_equal,
            self._base_theta,
            dtype=self.dtype,
            device=self.freqs[0].device,
        )
        with torch.no_grad():
            for param, init in zip(self.freqs, freqs):
                param.copy_(init)
