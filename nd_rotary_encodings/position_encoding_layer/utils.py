from enum import Enum
from typing import Union

import torch
from pytorch_sparse_utils.validation import validate_atleast_nd, validate_nd
from torch import Tensor


class FreqGroupPattern(Enum):
    SINGLE = "single"
    PARTITION = "partition"
    CLOSURE = "closure"

    def __str__(self):
        return self.value


def _validate_head_dim_even(head_dim: int):
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")


@torch.jit.ignore  # pyright: ignore[reportArgumentType]
def can_broadcast_shapes(*shapes) -> bool:
    """
    Returns True if the shapes can be broadcasted.

    Args:
        *shapes: Shapes to check.

    Returns:
        bool: True if shapes can be broadcasted, False if not
    """
    # Handle case where shapes is a single list of shapes
    if (
        len(shapes) == 1
        and hasattr(shapes[0], "__iter__")
        and not isinstance(shapes[0], (torch.Size, tuple))
    ):
        shapes = shapes[0]

    try:
        torch.broadcast_shapes(*shapes)
        return True
    except RuntimeError:
        return False


def prep_multilevel_positions(
    spatial_positions: Tensor,
    batch_indices: Tensor,
    level_indices: Tensor,
    level_spatial_shapes: Tensor,
):
    """Standardizes positional coordinates across multiple resolution levels.

    Converts indices or positions from multiple resolution levels to a standardized
    coordinate system by rescaling each level to match the finest level's resolution.
    This enables consistent position encoding across hierarchical feature maps.

    Args:
        spatial_positions (Tensor): Indices or positions of shape [..., position_dim],
            where each row contains the N-D position of each point. If floating point,
            they're treated as coordinates; if integer, they're treated as indices.
        batch_indices (Tensor): Integer tensor of shape [...], containing the
            batch index for each position in spatial_positions.
        level_indices (Tensor): Integer tensor of shape [...], containing the
            level index for each position in spatial_positions.
        level_spatial_shapes (Tensor): Tensor of shape [num_levels, 2] or
            [batch_size, num_levels, 2] specifying the spatial dimensions
            (height, width) of each level.

    Returns:
        Tensor: Rescaled positions of shape [..., position_dim + 1] with floating
            point dtype, where the last dimension has the level index concatenated onto
            the end of the spatial coordinates, and the spatial coordinates are
            standardized to the finest resolution level.

    Raises:
        ValueError: If tensors don't have the expected shape, dimensions, or dtypes.
    """
    validate_atleast_nd(spatial_positions, 2, "spatial_positions")
    batch_dims = spatial_positions.ndim - 1
    validate_nd(batch_indices, batch_dims, "batch_indices")
    validate_nd(level_indices, batch_dims, "level_indices")

    if not torch.is_floating_point(spatial_positions):
        # convert from indices to coordinates of pixel centers
        spatial_positions = spatial_positions + 0.5

    # batch, level, pos_dim or level, pos_dim
    assert level_spatial_shapes.ndim in (2, 3)

    # Initialize output tensor
    multilevel_positions = spatial_positions.new_zeros(
        spatial_positions.shape[:-1] + (spatial_positions.size(-1) + 1,)
    )

    # Early exit
    if multilevel_positions.numel() == 0:
        return multilevel_positions

    if level_spatial_shapes.ndim == 2:
        level_spatial_shapes = level_spatial_shapes.unsqueeze(0).expand(
            int(torch.max(batch_indices).item()) + 1, -1, -1
        )

    batch_max_spatial_shape = level_spatial_shapes.max(-2)[0]
    max_spatial_shapes = batch_max_spatial_shape[batch_indices]
    indexed_spatial_shapes = level_spatial_shapes[batch_indices, level_indices]

    # Fill in rescaled positions
    multilevel_positions[..., :-1] = spatial_positions / (
        indexed_spatial_shapes / max_spatial_shapes
    )

    # Fill in level indices
    multilevel_positions[..., -1] = level_indices.to(multilevel_positions)

    return multilevel_positions


def get_multilevel_freq_group_pattern(
    position_dim: int, pattern_name: Union[str, FreqGroupPattern], device=None
) -> Tensor:
    """Get a predefined frequency group pattern for RoPE encodings of multilevel features.

    Creates a frequency group pattern tensor for use with RoPEEncodingND based on
    predefined patterns that determine how spatial and level dimensions are encoded.

    Args:
        position_dim (int): Spatial dimension of the features to be encoded (2 for 2D
            images, etc.). The output tensor will have this many spatial dimensions
            plus 1 dimension for the feature level
        pattern_name (Union[str, FreqGroupPattern]): Pattern to use, either as a string
            or enum value. Options:
            - "single" or FreqGroupPattern.SINGLE: All dimensions (*spatial, level) in
                a single frequency group
            - "partition" or FreqGroupPattern.PARTITION: Spatial dimensions and level
                in separate groups
            - "closure" or FreqGroupPattern.CLOSURE: Three groups - Spatial, level,
                and (*spatial, level)
        device (torch.device, optional): Device for the created tensor. Defaults to None.

    Returns:
        Tensor: Boolean tensor encoding the frequency group pattern, of shape
            [n_freq_groups, position_dim + 1]

    Raises:
        ValueError: If an unrecognized pattern name is provided.
    """
    if isinstance(pattern_name, FreqGroupPattern):
        pattern_name = pattern_name.value

    if pattern_name == "single":
        out = torch.ones(1, position_dim + 1, device=device)
    elif pattern_name == "partition":
        out = torch.zeros(2, position_dim + 1, device=device)
        out[0, :-1] = True  # Spatial dimensions in one group
        out[1, -1] = True  # Level dimension in second group
    elif pattern_name == "closure":
        out = torch.zeros(3, position_dim + 1, device=device)
        out[0, :-1] = True  # Spatial dimensions in one group
        out[1, -1] = True  # Level dimension in second group
        out[2, :] = True  # Third group has all dimensions
    else:
        raise ValueError(f"Unrecognized pattern_name {pattern_name}")

    return out
