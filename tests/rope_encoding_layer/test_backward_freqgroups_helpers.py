from math import prod
from typing import Any

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings

from nd_rotary_encodings import (
    RoPEEncodingND,
)
from nd_rotary_encodings.functional.forward_backward_fns import (
    calculate_rope,
    rotate_embeddings,
)

from .conftest import rope_config_strategy, rope_input_tensors


@pytest.mark.cuda_if_available
class TestRoPEEncodingNDBackward:
    def test_backward_basic(
        self, base_config: dict[str, Any], sample_data, device: str
    ):
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        query = sample_data["query"].requires_grad_(True)
        query_pos = sample_data["query_pos"].requires_grad_(True)
        key = sample_data["key"].requires_grad_(True)
        key_pos = sample_data["key_pos"].requires_grad_(True)

        query_rotated, key_rotated = rope(query, query_pos, key, key_pos)

        loss = query_rotated.sum() + key_rotated.sum()
        loss.backward()

        assert query.grad is not None
        assert query_pos.grad is not None
        assert key.grad is not None
        assert key_pos.grad is not None

    def test_backward_gradcheck(
        self, base_config: dict[str, Any], sample_data, device: str
    ):
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=torch.double,
        ).to(device)

        query = sample_data["query"].double().normal_().requires_grad_(True)
        query_pos = sample_data["query_pos"].double().normal_().requires_grad_(True)
        key = sample_data["key"].double().normal_().requires_grad_(True)
        key_pos = sample_data["key_pos"].double().normal_().requires_grad_(True)

        assert torch.autograd.gradcheck(rope, (query, query_pos, key, key_pos))

    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    @given(config=rope_config_strategy(require_grads=True))
    def test_backward_hypothesis(self, config, device: str):

        assume(
            config["tensor_config"]["query_require_grad"]
            or config["tensor_config"]["query_pos_require_grad"]
            or config["tensor_config"]["key_require_grad"]
            or config["tensor_config"]["key_pos_require_grad"]
        )

        rope = RoPEEncodingND(**config["config"]).to(device)
        input_tensors = rope_input_tensors(**config["tensor_config"], device=device)

        output = rope(**input_tensors)
        if isinstance(output, tuple):
            query_rotated, key_rotated = output
            loss = query_rotated.sum() + key_rotated.sum()
        else:
            query_rotated = output
            loss = query_rotated.sum()

        loss.backward()

        if config["tensor_config"]["query_require_grad"]:
            assert input_tensors["query"].grad is not None
        if config["tensor_config"]["query_pos_require_grad"]:
            assert input_tensors["query_pos"].grad is not None
        if config["tensor_config"]["key_require_grad"]:
            assert input_tensors["key"].grad is not None
        if config["tensor_config"]["key_pos_require_grad"]:
            assert input_tensors["key_pos"].grad is not None

    @pytest.mark.filterwarnings("ignore:^Expected un-normalized:UserWarning")
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.differing_executors],
        max_examples=25,
    )
    @given(config=rope_config_strategy(require_grads=True))
    def test_gradcheck_hypothesis(self, config, device: str):
        assume(not config["config"]["forward_only"])
        assume(
            config["tensor_config"]["query_require_grad"]
            or config["tensor_config"]["query_pos_require_grad"]
            or config["tensor_config"]["key_require_grad"]
            or config["tensor_config"]["key_pos_require_grad"]
        )

        assume(prod(config["tensor_config"]["batch_shape_query"]) <= 100)
        if config["tensor_config"]["batch_shape_key"] is not None:
            assume(prod(config["tensor_config"]["batch_shape_key"]) <= 100)

        config["config"]["dtype"] = torch.double

        rope = RoPEEncodingND(**config["config"]).to(device)
        input_tensors = rope_input_tensors(**config["tensor_config"], device=device)

        max_val = 0.0

        query = input_tensors["query"].double().clamp(-1e4, 1e4)
        if query.requires_grad and query.numel() > 0:
            max_val = max(max_val, query.abs().max().item())

        query_pos = input_tensors["query_pos"]
        if torch.is_floating_point(query_pos):
            query_pos = query_pos.double().clamp(-1e4, 1e4)
            if query_pos.requires_grad and query_pos.numel() > 0:
                max_val = max(max_val, query_pos.abs().max().item())

        key = input_tensors["key"]
        if key is not None:
            key = key.double().clamp(-1e4, 1e4)
            if key.requires_grad and key.numel() > 0:
                max_val = max(max_val, key.abs().max().item())

        key_pos = input_tensors["key_pos"]
        if key_pos is not None and torch.is_floating_point(key_pos):
            key_pos = key_pos.double().clamp(-1e4, 1e4)
            if key_pos.requires_grad and key_pos.numel() > 0:
                max_val = max(max_val, key_pos.abs().max().item())

        if max_val > 1e6:
            rtol = 0.01
        else:
            rtol = 0.001

        assert torch.autograd.gradcheck(
            rope, (query, query_pos, key, key_pos), rtol=rtol
        )


@pytest.mark.cuda_if_available
class TestRoPEEncodingNDFrequencyGroups:
    """Tests specifically for frequency group functionality."""

    @pytest.mark.parametrize(
        "n_freq_groups,enforce_equal",
        [
            (1, True),  # Single group (default)
            (2, True),  # Two equal groups
            (4, True),  # Four equal groups
            (3, False),  # Three groups, may not be equal
        ],
    )
    def test_freq_group_dimension_distribution(
        self, n_freq_groups, enforce_equal, base_config: dict[str, Any], device: str
    ):
        """Test different frequency group configurations and dimension distribution."""
        # Create pattern with one position dimension per group, cycling if needed
        position_dim = max(base_config["position_dim"], n_freq_groups)
        freq_group_pattern = torch.zeros(
            n_freq_groups, position_dim, dtype=torch.bool, device=device
        )
        for g in range(n_freq_groups):
            freq_group_pattern[g, g % position_dim] = True

        # Adjust embed_dim to ensure it's cleanly divisible when enforce_equal=True
        embed_dim = base_config["embed_dim"]
        head_dim = embed_dim // base_config["n_heads"]

        if enforce_equal and (head_dim // 2) % n_freq_groups != 0:
            # Adjust embed_dim to make head_dim/2 divisible by n_freq_groups
            head_dim = ((head_dim // 2) // n_freq_groups) * n_freq_groups * 2
            if head_dim == 0:
                head_dim = n_freq_groups * 2  # minimum valid head_dim
            embed_dim = head_dim * base_config["n_heads"]

        # Initialize RoPE
        rope = RoPEEncodingND(
            position_dim=position_dim,
            embed_dim=embed_dim,
            n_heads=base_config["n_heads"],
            freq_group_pattern=freq_group_pattern,
            enforce_freq_groups_equal=enforce_equal,
            dtype=base_config["dtype"],
        ).to(device)

        # Verify the number of frequency groups
        assert len(rope.freqs) == n_freq_groups

        # Check encoding dimensions
        half_head_dim = rope.head_dim // 2

        if enforce_equal:
            # All groups should have the same number of encoding dimensions
            expected_dims_per_group = half_head_dim // n_freq_groups
            for freq in rope.freqs:
                assert freq.shape[2] == expected_dims_per_group
        else:
            # Sum of dimensions should equal half_head_dim
            total_dims = sum(freq.shape[2] for freq in rope.freqs)
            assert total_dims == half_head_dim

            # If not enforcing equality, earlier groups may have more dimensions
            if n_freq_groups > 1 and half_head_dim % n_freq_groups != 0:
                # First group should have more dimensions than last
                assert rope.freqs[0].shape[2] >= rope.freqs[-1].shape[2]

        # Verify encoding ranges
        assert rope.encoding_ranges.shape == (n_freq_groups, 2)
        assert (
            rope.encoding_ranges[-1, 1] == half_head_dim
        )  # Last end should be half_head_dim

        # Check that ranges are contiguous
        for i in range(n_freq_groups - 1):
            assert rope.encoding_ranges[i, 1] == rope.encoding_ranges[i + 1, 0]

    def test_grouped_rope_freqs_tensor(self, base_config: dict[str, Any], device: str):
        """Test the grouped_rope_freqs_tensor method."""
        position_dim = 3
        n_freq_groups = 2

        # Create a custom pattern
        freq_group_pattern = torch.tensor(
            [
                [True, True, False],  # Group 1: dims 0,1
                [False, False, True],  # Group 2: dim 2
            ],
            device=device,
        )

        rope = RoPEEncodingND(
            position_dim=position_dim,
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            freq_group_pattern=freq_group_pattern,
            dtype=base_config["dtype"],
        ).to(device)

        # Get the frequency tensor
        freq_tensor = rope.grouped_rope_freqs_tensor(rope.freqs)

        # Check shape
        expected_shape = (
            position_dim,
            n_freq_groups,
            base_config["n_heads"],
            rope.head_dim // 2,
        )
        assert freq_tensor.shape == expected_shape

        # Check zeros in appropriate places (where freq_group_pattern is False)
        for dim in range(position_dim):
            for group in range(n_freq_groups):
                if not freq_group_pattern[group, dim]:
                    assert torch.all(freq_tensor[dim, group] == 0)

    def test_grouped_rope_freqs_tensor_implementation_equivalence(
        self, base_config: dict[str, Any], device: str
    ):
        """Test that current grouped_rope_freqs_tensor implementation matches the old
        version that uses direct indexing."""
        # Setup test parameters with mixed frequency groups for a thorough test
        position_dim = 3

        # Create a custom pattern where each group handles different position dimensions
        freq_group_pattern = torch.tensor(
            [
                [True, True, False],  # Group 1: dims 0,1
                [False, False, True],  # Group 2: dim 2
            ],
            device=device,
        )

        rope = RoPEEncodingND(
            position_dim=position_dim,
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            freq_group_pattern=freq_group_pattern,
            dtype=base_config["dtype"],
        ).to(device)

        # Define the old implementation
        def old_grouped_rope_freqs_tensor(rope_instance, grouped_rope_freqs):
            if isinstance(grouped_rope_freqs, torch.Tensor):
                grouped_rope_freqs = [grouped_rope_freqs]

            n_heads = rope_instance.n_heads if not rope_instance.share_heads else 1
            rope_freqs = grouped_rope_freqs[0].new_zeros(
                rope_instance.n_freq_groups,
                rope_instance.position_dim,
                n_heads,
                rope_instance.head_dim // 2,
            )

            freq_group_pattern = rope_instance.freq_group_pattern
            for g, (freqs_g, range_g) in enumerate(
                zip(grouped_rope_freqs, rope_instance.encoding_ranges)
            ):
                range_start, range_end = range_g
                rope_freqs[g, freq_group_pattern[g], :, range_start:range_end] = freqs_g

            # Transpose to output shape
            rope_freqs = rope_freqs.transpose(0, 1).contiguous()
            return rope_freqs

        # Get results from both implementations
        current_result = rope.grouped_rope_freqs_tensor(rope.freqs)
        old_result = old_grouped_rope_freqs_tensor(rope, rope.freqs)

        # Compare results
        assert current_result.shape == old_result.shape, (
            f"Shape mismatch: current {current_result.shape}, "
            f"old {old_result.shape}"
        )

        assert torch.allclose(current_result, old_result), (
            "Values differ between implementations. Max diff: "
            f"{(current_result - old_result).abs().max()}"
        )

        # Test with both standard and shared heads configurations
        rope_shared = RoPEEncodingND(
            position_dim=position_dim,
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            freq_group_pattern=freq_group_pattern,
            share_heads=True,
            dtype=base_config["dtype"],
        ).to(device)

        current_result_shared = rope_shared.grouped_rope_freqs_tensor(rope_shared.freqs)
        old_result_shared = old_grouped_rope_freqs_tensor(
            rope_shared, rope_shared.freqs
        )

        assert torch.allclose(
            current_result_shared, old_result_shared
        ), "Implementations produce different results with shared heads"

    @settings(deadline=None)
    @given(inputs=rope_config_strategy())
    def test_grouped_rope_freqs_tensor_hypothesis(
        self, inputs: dict[str, Any], device: str
    ):
        """Test structural properties of the grouped_rope_freqs_tensor method."""
        rope = RoPEEncodingND(**inputs["config"]).to(device)

        # Get the full frequency tensor
        freq_tensor = rope.grouped_rope_freqs_tensor(rope.freqs)

        # Verify shape
        n_heads = 1 if rope.share_heads else rope.n_heads
        expected_shape = (
            rope.position_dim,
            rope.n_freq_groups,
            n_heads,
            rope.head_dim // 2,
        )
        assert freq_tensor.shape == expected_shape

        # Verify placement of frequencies
        for g, freq_g in enumerate(rope.freqs):
            range_start, range_end = rope.encoding_ranges[g]
            pos_dims = torch.nonzero(rope.freq_group_pattern[g], as_tuple=True)[0]

            for i, pos_dim in enumerate(pos_dims):
                extracted = freq_tensor[pos_dim, g, :, range_start:range_end]
                expected = freq_g[i, :, :]
                assert torch.allclose(extracted, expected)

        # Verify zeros where expected
        for g in range(rope.n_freq_groups):
            for pos_dim in range(rope.position_dim):
                if not rope.freq_group_pattern[g, pos_dim]:
                    # This position dimension should be zero in this frequency group
                    assert torch.all(freq_tensor[pos_dim, g] == 0)

        # Verify encoding ranges are contiguous and complete
        assert rope.encoding_ranges[0, 0] == 0
        assert rope.encoding_ranges[-1, 1] == rope.head_dim // 2
        for i in range(len(rope.encoding_ranges) - 1):
            assert rope.encoding_ranges[i, 1] == rope.encoding_ranges[i + 1, 0]

        # Verify encoding dimension distribution
        if rope.n_freq_groups > 1:
            encoding_sizes = [
                (end - start).item() for start, end in rope.encoding_ranges
            ]
            if rope.enforce_freq_groups_equal:
                # All groups should have equal encoding dimensions
                assert len(set(encoding_sizes)) == 1
            else:
                # For unequal distribution, the difference should be at most 1
                assert max(encoding_sizes) - min(encoding_sizes) <= 1

        # Verify all values are finite
        assert torch.all(torch.isfinite(freq_tensor))

        # Verify dtype and device match input
        assert freq_tensor.dtype == rope.freqs[0].dtype
        assert freq_tensor.device == rope.freqs[0].device

    @settings(deadline=None)
    @given(inputs=rope_config_strategy())
    def test_grouped_rope_freqs_tensor_hypothesis_math(
        self, inputs: dict[str, Any], device: str
    ):
        """Test mathematical properties of the grouped_rope_freqs_tensor method."""
        rope = RoPEEncodingND(**inputs["config"]).to(device)
        freq_tensor = rope.grouped_rope_freqs_tensor(rope.freqs)
        n_heads = 1 if rope.share_heads else rope.n_heads

        # Check magnitudes are monotonically decreasing
        if rope.freqs[0].numel() > 0:
            for g, ranges in enumerate(rope.encoding_ranges):
                range_size = ranges[1] - ranges[0]
                if range_size <= 1:
                    continue  # Skip if not enough dimensions to compare

                # Find any non-zero position dimension for this group
                pos_dims = torch.nonzero(rope.freq_group_pattern[g], as_tuple=True)[0]
                if len(pos_dims) == 0:
                    continue

                for pos_dim in pos_dims:
                    for head_idx in range(n_heads):
                        pos_dim = pos_dims[0]
                        head_idx = 0

                        # Extract frequencies and take absolute values
                        freqs = freq_tensor[pos_dim, g, head_idx, ranges[0] : ranges[1]]
                        magnitudes = torch.abs(freqs)

                        if magnitudes.numel() >= 3:
                            diffs = magnitudes[1:] - magnitudes[:-1]

                            # Should be monotonic
                            assert torch.all(diffs <= 0), (
                                "Expected consistent geometric progression "
                                "in frequency magnitudes"
                            )


@pytest.mark.cuda_if_available
class TestRoPEEncodingNDHelperMethods:
    """Tests for the helper methods of RoPEEncodingND."""

    def test_position_grid(self, device: str):
        """Test the position_grid static method."""
        # Test 2D grid
        shape_2d = (1, 3, 4, 256)  # batch, height, width, features
        grid_2d = RoPEEncodingND.position_grid(shape_2d, device=device)

        # Check shape: should be [height, width, 2]
        assert grid_2d.shape == (3, 4, 2)

        # Check values: grid should contain coordinates
        for h in range(3):
            for w in range(4):
                assert torch.equal(grid_2d[h, w], torch.tensor([h, w], device=device))

        # Test 3D grid
        shape_3d = (1, 2, 3, 4, 256)  # batch, depth, height, width, features
        grid_3d = RoPEEncodingND.position_grid(shape_3d, device=device)

        # Check shape: should be [depth, height, width, 3]
        assert grid_3d.shape == (2, 3, 4, 3)

        # Check values: grid should contain coordinates
        for d in range(2):
            for h in range(3):
                for w in range(4):
                    assert torch.equal(
                        grid_3d[d, h, w], torch.tensor([d, h, w], device=device)
                    )

    def test_calculate_rope(self, base_config: dict[str, Any], device: str):
        """Test the calculate_rope static method."""
        position_dim = base_config["position_dim"]
        n_heads = base_config["n_heads"]
        head_dim = base_config["embed_dim"] // n_heads

        # Create positions and frequencies
        positions = torch.rand(2, 10, position_dim, device=device) * 10
        rope_freqs = torch.rand(position_dim, 1, n_heads, head_dim // 2, device=device)

        # Calculate rope encodings
        rope_encodings = RoPEEncodingND.calculate_rope(positions, rope_freqs)

        # Check shape
        assert rope_encodings.shape == (2, 10, n_heads, head_dim // 2)

        # Verify it matches the imported function
        expected = calculate_rope(positions, rope_freqs)
        assert torch.allclose(rope_encodings, expected)

    def test_rotate_embeddings(self, base_config: dict[str, Any], device: str):
        """Test the rotate_embeddings static method."""
        n_heads = base_config["n_heads"]
        head_dim = base_config["embed_dim"] // n_heads

        # Create embeddings and encodings
        embeddings = torch.randn(2, 10, n_heads, head_dim, device=device)
        rope_encodings = torch.rand(2, 10, n_heads, head_dim // 2, device=device)

        # Rotate embeddings
        rotated = RoPEEncodingND.rotate_embeddings(embeddings, rope_encodings)

        # Check shape
        assert rotated.shape == embeddings.shape

        # Verify it matches the imported function
        expected = rotate_embeddings(embeddings, rope_encodings)
        assert torch.allclose(rotated, expected)

    def test_reset_parameters(self, base_config: dict[str, Any], device: str):
        """Test the reset_parameters method."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        # Store initial parameters
        initial_params = [param.clone() for param in rope.freqs]

        # Reset parameters
        rope.reset_parameters()

        # Check if parameters changed
        any_changed = False
        for old_param, new_param in zip(initial_params, rope.freqs):
            if not torch.allclose(old_param, new_param):
                any_changed = True
                break

        # Parameters should have changed (very unlikely they'd be identical)
        assert any_changed, "Expected parameters to change after reset"
