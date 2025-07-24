import torch
from typing import Optional, Union
from torch import Tensor

from .utils import validate_head_dim_even

def init_2d_freqs_rope_mixed_orig(
    head_dim: int,
    num_heads: int,
    theta: float = 10.0,
    rotate: bool = True,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Slightly modified version of the original RoPE-Mixed initialization function."""
    freqs_x = []
    freqs_y = []
    mag = 1 / (
        theta ** (torch.arange(0, head_dim, 4, dtype=dtype, device=device) / head_dim)
    )
    for _ in range(num_heads):
        angles = (
            torch.rand(1, device=device, dtype=dtype) * 2 * torch.pi
            if rotate
            else torch.zeros(1, device=device, dtype=dtype)
        )
        fx = torch.cat(
            [mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1
        )
        fy = torch.cat(
            [mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=-1)
    return freqs  # n_head, head_dim/2, 2


def init_2d_freqs_rope_mixed(
    head_dim: int,
    n_heads: int,
    theta: float = 10.0,
    rotate: bool = True,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tensor:
    """Initializes frequency parameters for 2D rotary position embeddings.

    Generates the frequencies used for RoPE (Rotary Position Embeddings) in two dimensions.
    For each head, creates frequency vectors that incorporate both magnitude decay based
    on theta and optional random per-head rotation angles.

    Args:
        head_dim (int): Dimension size of each attention head. Must be divisible by 2.
        n_heads (int): Number of attention heads.
        theta (float): Base value for frequency scaling. Larger values result in longer
            period sinusoids. Default: 10.0
        rotate (bool): Whether to apply random rotation to the frequency vectors.
            When True, each head gets different random rotations. Default: True
        dtype (Optional[torch.dtype]): Data type for the output tensor. Default: None
        device (Optional[str | torch.device]): Device for the output tensor. Default: None

    Returns:
        Tensor: Frequency parameter tensor of shape [2, n_heads, head_dim/2], containing
            the frequency parameters for x and y dimensions for each attention head.

    Raises:
        ValueError: If head_dim is not divisible by 2.
    """
    validate_head_dim_even(head_dim)

    # Create frequency magnitudes that decay with head_dim index
    dim_t = torch.arange(0, head_dim, 2, dtype=dtype, device=device)
    dim_t = theta ** (dim_t / head_dim)

    freqs = torch.zeros(2, n_heads, head_dim // 2, device=device, dtype=dtype)
    for dim_index in range(freqs.size(0)):
        for head_index in range(n_heads):
            angle = (
                torch.rand(1, device=device, dtype=dtype) * 2 * torch.pi
                if rotate
                else torch.zeros(1, device=device, dtype=dtype)
            )
            head_freqs = torch.cos(angle) * dim_t  # shape: [head_dim / 2]
            freqs[dim_index, head_index, :] = head_freqs

    return freqs


def init_nd_freqs(
    position_dim: int,
    head_dim: int,
    num_heads: int,
    freq_group_pattern: Tensor,
    enforce_freq_groups_equal: bool = True,
    thetas: Union[Tensor, float, list[list[float]]] = 10.0,
    rotate: bool = True,
    max_rotation_angle: float = 2 * torch.pi,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> tuple[list[Tensor], Tensor]:
    """Initializes frequency parameters for N-dimensional rotary position embeddings.

    Generates the frequencies used for RoPE (Rotary Position Embeddings) in N dimensions.
    For each head and position dimension, creates frequency vectors with magnitude decay
    based on theta values and optional random rotation angles.

    Args:
        position_dim (int): Number of position dimensions (e.g., 2 for 2D, 3 for 3D).
        head_dim (int): Dimension size of each attention head. Must be divisible by 2.
        num_heads (int): Number of attention heads. Can be 1 even if the embeddings to
            be encoded will be split into multiple heads to share RoPE encodings among
            heads.
        freq_group_pattern (Tensor): Boolean tensor of shape
            [n_freq_groups, position_dim] defining frequency group inclusion. The
            (head_dim/2) elements of the RoPE encoding vector will be split among the
            frequency groups. If position i, j is True, then frequency group i
            includes position dimension j.
        enforce_freq_groups_equal (boolean): If True, then this function will raise
            a ValueError if the (head_dim/2) available elements of the RoPE vector
            cannot be evenly split between the frequency groups. If False, then
            trailing frequency groups may have fewer RoPE encodings assigned to them.
        thetas (Union[Tensor], float, list[list[float]]]): Base value(s) for frequency
            scaling. Can be a single float (applied to all dimensions and frequency
            groups) or a 2D tensor of shape [n_freq_groups, position_dim], with either
            component dimension allowed to be 1 for broadcasting. Entries corresponding
            to non-included position dimensions in a frequency group will be ignored.
            Larger values of theta result in lower-frequency rotations, and may be more
            suitable for dimensions of greater spatial scale. Default: 10.0
        rotate (bool): Whether to apply random rotation to the frequency vectors. When True,
            each head gets different random rotations. Default: True
        max_rotation_angle (bool): If rotate is True, each head's random rotation is
            uniformly distributed between 0 and max_rotation_angle. Default: 2 * pi
        dtype (Optional[torch.dtype]): Data type for the output tensor. Default: None
        device (Optional[str | torch.device]): Device for the output tensor. If None, the
            tensor will be created on freq_group_pattern's device. Default: None

    Returns:
        list[Tensor]: n_freq_groups-long list of frequency tensors, each of shape
            [position_dim_g, n_heads, head_dim//(2 * n_freq_groups)] (or
            [position_dim_g, n_heads, head_dim//(2 * n_freq_groups) + 1], if
            enforce_greq_groups_equal is False and some frequency groups have fewer
            encoding dimensions), where position_dim_g is the number of position
            dimensions included in that frequency group (i.e.,
            freq_group_pattern[g].sum()), containing the frequency parameters for each
            position dimension and attention head.
        Tensor: Long tensor of shape [n_freq_groups, 2] of the start and end indices of
            the RoPE encoding dimensions that are assigned to each frequency group

    Raises:
        ValueError: If head_dim is not divisible by 2, if freq_group pattern is not
        2D or has second dimension size not equal to position_dim, if
        enforce_freq_groups_equal is True and (head_dim/2) is not evenly divisible
        by the number of frequency groups, or if thetas is the wrong size.

    Notes:
        Differences from rope-for-vit:
            - Decreasing frequencies over encoding dim instead of increasing
                (theta raised to negative power instead of positive) - similar to
                standard 1D RoPE
            - Configurable max rotation angle
    """
    validate_head_dim_even(head_dim)

    if device is None:
        device = freq_group_pattern.device

    n_freq_groups = freq_group_pattern.size(0)

    # Validate thetas (base frequencies)
    thetas = torch.as_tensor(thetas, dtype=dtype, device=device)
    if thetas.ndim != 2 and thetas.numel() != 1:
        raise ValueError(
            "Expected thetas to either be a scalar or a 2D tensor, got shape "
            f"{thetas.shape}."
        )

    # broadcast thetas
    if thetas.numel() == 1 or 1 in thetas.shape:
        thetas = thetas.expand(n_freq_groups, position_dim)

    if thetas.shape != (n_freq_groups, position_dim):
        raise ValueError(
            "Expected thetas to be broadcastable to [n_freq_groups, position_dim] "
            f"([{n_freq_groups}, {position_dim}]), got shape {thetas.shape}"
        )

    # Assign RoPE encoding dims to frequency groups
    half_head_dim = head_dim // 2
    base_dim = half_head_dim // n_freq_groups
    remainder = half_head_dim % n_freq_groups

    if remainder > 0 and enforce_freq_groups_equal:
        raise ValueError(
            f"RoPE encodings ({half_head_dim}) not evenly divisible by frequency "
            f"groups ({n_freq_groups})"
        )

    # Create tensor with base dimensions and add remainder to first elements
    encodings_per_freq_group = torch.full((n_freq_groups,), base_dim, device=device)
    encodings_per_freq_group[:remainder] += 1

    # Initialize grouped RoPE frequencies
    freqs = []
    encoding_ranges: list[tuple[int, int]] = []
    encoding_start = 0
    for g in range(n_freq_groups):
        freq_group_size = int(encodings_per_freq_group[g].item())
        n_pos_dims_this_freq_group = int(freq_group_pattern[g].sum().item())
        freqs_g = torch.zeros(
            (n_pos_dims_this_freq_group, num_heads, freq_group_size),
            dtype=dtype,
            device=device,
        )

        encoding_ranges.append((encoding_start, encoding_start + freq_group_size))
        encoding_start = encoding_start + freq_group_size

        group_dim_counter = 0
        # loop over all position dims, skipping excluded ones for this freq group
        for dim_index in range(position_dim):
            if not freq_group_pattern[g, dim_index]:
                continue
            theta_g_dim = thetas[g, dim_index]

            # Create frequency magnitudes that decay with RoPE encoding index
            dim_t = torch.arange(0, freq_group_size, 1, dtype=dtype, device=device)
            dim_t = theta_g_dim ** (-dim_t / freq_group_size)

            for head_index in range(num_heads):
                angle = (
                    torch.rand(1, device=device, dtype=dtype) * max_rotation_angle
                    if rotate
                    else torch.zeros(1, device=device, dtype=dtype)
                )
                head_freqs = torch.cos(angle) * dim_t
                freqs_g[group_dim_counter, head_index, :] = head_freqs
            group_dim_counter += 1

        freqs.append(freqs_g)

    encoding_ranges_t = torch.tensor(encoding_ranges, dtype=torch.long, device=device)
    return freqs, encoding_ranges_t
