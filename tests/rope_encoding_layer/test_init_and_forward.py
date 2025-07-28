from math import isclose
from typing import Any

import hypothesis
import pytest
import torch
from hypothesis import given, settings

from nd_rotary_encodings import (
    RoPEEncodingND,
)

from .conftest import rope_config_strategy, rope_input_tensors


@pytest.mark.cuda_if_available
class TestRoPEEncodingNDInitialization:
    """Tests for RoPEEncodingND initialization with various configurations."""

    def test_basic_initialization(self, base_config: dict[str, Any], device: str):
        """Test basic initialization with default parameters."""
        rope = RoPEEncodingND(**base_config).to(device)

        assert rope.position_dim == base_config["position_dim"]
        assert rope.embed_dim == base_config["embed_dim"]
        assert rope.n_heads == base_config["n_heads"]
        assert rope.head_dim == base_config["embed_dim"] // base_config["n_heads"]
        assert rope.dtype == base_config["dtype"]
        assert len(rope.freqs) == 1  # Default is 1 frequency group
        assert rope.freqs[0].shape == (
            base_config["position_dim"],
            base_config["n_heads"],
            rope.head_dim // 2,
        )
        assert rope.freq_group_pattern.shape == (1, base_config["position_dim"])
        assert torch.all(rope.freq_group_pattern)  # All True

    @pytest.mark.parametrize("position_dim", [1, 2, 3, 4])
    def test_different_dimensions(
        self, position_dim, base_config: dict[str, Any], device: str
    ):
        """Test initialization with different position dimensions."""
        rope = RoPEEncodingND(
            position_dim=position_dim,
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        assert rope.position_dim == position_dim
        assert rope.freq_group_pattern.shape == (1, position_dim)
        assert rope.freqs[0].shape == (
            position_dim,
            base_config["n_heads"],
            rope.head_dim // 2,
        )

    @pytest.mark.parametrize("rope_base_theta", [10.0, 100.0, 1000.0])
    def test_different_theta(
        self, rope_base_theta, base_config: dict[str, Any], device: str
    ):
        """Test initialization with different theta values."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            rope_base_theta=rope_base_theta,
            dtype=base_config["dtype"],
        ).to(device)

        # Check that the base theta is stored
        assert isclose(float(rope._base_theta), rope_base_theta)

        # Generate two models with different theta
        if rope_base_theta == 10.0:  # Only need to compare once
            rope2 = RoPEEncodingND(
                position_dim=base_config["position_dim"],
                embed_dim=base_config["embed_dim"],
                n_heads=base_config["n_heads"],
                rope_base_theta=rope_base_theta * 10,  # Different theta
                dtype=base_config["dtype"],
            ).to(device)

            # Compare frequencies - they should be different
            assert not torch.allclose(rope.freqs[0], rope2.freqs[0])

    def test_custom_frequency_pattern(self, base_config: dict[str, Any], device: str):
        """Test initialization with custom frequency group patterns."""
        position_dim = 3  # Use 3 dimensions for this test

        # Create a custom pattern with 2 frequency groups
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

        # Verify the pattern was stored correctly
        assert torch.equal(rope.freq_group_pattern, freq_group_pattern)

        # Verify we have 2 frequency groups
        assert len(rope.freqs) == 2

        # First group should have 2 dimensions, second group should have 1
        assert rope.freqs[0].shape[0] == 2
        assert rope.freqs[1].shape[0] == 1

        # Verify encoding ranges
        head_dim_half = rope.head_dim // 2
        assert torch.equal(
            rope.encoding_ranges,
            torch.tensor(
                [[0, head_dim_half // 2], [head_dim_half // 2, head_dim_half]],
                device=device,
            ),
        )

    def test_initialization_errors(self, base_config: dict[str, Any], device: str):
        """Test initialization with invalid parameters triggers appropriate errors."""
        # Test odd embed_dim
        with pytest.raises(
            ValueError, match="Expected embed_dim to be divisible by n_heads"
        ):
            RoPEEncodingND(
                position_dim=base_config["position_dim"],
                embed_dim=base_config["embed_dim"] + 1,  # Odd number
                n_heads=base_config["n_heads"],
            ).to(device)

        # Test odd head_dim (embed_dim / n_heads)
        with pytest.raises(ValueError, match="Expected head_dim to be divisible by 2"):
            RoPEEncodingND(
                position_dim=base_config["position_dim"],
                embed_dim=base_config["n_heads"] * 3,  # Makes head_dim odd
                n_heads=base_config["n_heads"],
            ).to(device)

        # Test invalid frequency group pattern
        with pytest.raises(
            ValueError, match="Expected 2D tensor for freq_group_pattern"
        ):
            RoPEEncodingND(
                position_dim=base_config["position_dim"],
                embed_dim=base_config["embed_dim"],
                n_heads=base_config["n_heads"],
                freq_group_pattern=torch.ones(3, device=device),  # 1D tensor
            ).to(device)

        # Test mismatched position_dim and freq_group_pattern
        with pytest.raises(
            ValueError, match="Expected second dimension of freq_group_pattern"
        ):
            RoPEEncodingND(
                position_dim=base_config["position_dim"],
                embed_dim=base_config["embed_dim"],
                n_heads=base_config["n_heads"],
                freq_group_pattern=torch.ones(
                    1, base_config["position_dim"] + 1, device=device
                ),
            ).to(device)

        # Test bad optimization configs
        with pytest.raises(ValueError, match="`inplace=True` requires"):
            RoPEEncodingND(
                position_dim=base_config["position_dim"],
                embed_dim=base_config["embed_dim"],
                n_heads=base_config["n_heads"],
                forward_only=False,
                inplace=True,
            ).to(device)
        with pytest.raises(
            ValueError, match="`use_checkpointing` is incompatible with"
        ):
            RoPEEncodingND(
                position_dim=base_config["position_dim"],
                embed_dim=base_config["embed_dim"],
                n_heads=base_config["n_heads"],
                use_checkpointing=True,
                forward_only=True,
            )

    @settings(deadline=None)
    @given(inputs=rope_config_strategy())
    def test_initialization_hypothesis(self, inputs: dict[str, Any], device: str):
        config = inputs["config"]
        rope = RoPEEncodingND(**config).to(device)

        assert rope.position_dim == config["position_dim"]
        assert rope.embed_dim == config["embed_dim"]
        assert rope.n_heads == config["n_heads"]
        assert rope.head_dim == config["embed_dim"] // config["n_heads"]
        assert rope.dtype == config["dtype"]
        n_freq_groups = config["freq_group_pattern"].shape[0]
        assert config["freq_group_pattern"].shape[1] == config["position_dim"]
        assert len(rope.freqs) == n_freq_groups
        for i, freq in enumerate(rope.freqs):
            if config["enforce_freq_groups_equal"]:
                assert freq.shape == (
                    config["freq_group_pattern"][i].sum(),
                    config["n_heads"] if not config["share_heads"] else 1,
                    rope.head_dim // n_freq_groups // 2,
                )
            else:
                assert freq.shape[:-1] == (
                    config["freq_group_pattern"][i].sum(),
                    config["n_heads"] if not config["share_heads"] else 1,
                )


@pytest.mark.cuda_if_available
class TestRoPEEncodingNDForward:
    """Tests for the forward method with different input configurations."""

    def test_forward_query_only(
        self, base_config: dict[str, Any], sample_data, device: str
    ):
        """Test forward pass with only query tensor."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        query_rotated = rope(sample_data["query"], sample_data["query_pos"])

        # Verify shape is preserved
        assert query_rotated.shape == sample_data["query"].shape

        # Verify output is different from input
        assert not torch.allclose(query_rotated, sample_data["query"])

    @settings(deadline=None)
    @given(inputs=rope_config_strategy())
    def test_forward_hypothesis(self, inputs: dict[str, Any], device: str):
        """Test forward pass with Hypothesis-generated configs"""
        rope = RoPEEncodingND(**inputs["config"]).to(device)
        input_tensors = rope_input_tensors(**inputs["tensor_config"], device=device)

        query = input_tensors["query"]
        query_pos = input_tensors["query_pos"]
        key = input_tensors["key"]
        key_pos = input_tensors["key_pos"]

        hypothesis.assume(
            not torch.allclose(query_pos, torch.zeros_like(query_pos), atol=1e-4)
        )
        hypothesis.assume(not torch.allclose(query, torch.zeros_like(query), atol=1e-4))
        hypothesis.assume(
            key is None or not torch.allclose(key, torch.zeros_like(key), atol=1e-4)
        )
        hypothesis.assume(
            key_pos is None
            or not torch.allclose(key_pos, torch.zeros_like(key_pos), atol=1e-4)
        )

        output = rope(**input_tensors)
        if key is not None:
            query_rotated, key_rotated = output
        else:
            query_rotated = output

        # Test that rotation happened
        assert not torch.allclose(query, query_rotated)
        if key is not None:
            assert not torch.allclose(key, key_rotated)

    def test_forward_query_and_key(
        self, base_config: dict[str, Any], sample_data, device: str
    ):
        """Test forward pass with both query and key tensors."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        query_rotated, key_rotated = rope(
            sample_data["query"],
            sample_data["query_pos"],
            sample_data["key"],
            sample_data["key_pos"],
        )

        # Verify shapes are preserved
        assert query_rotated.shape == sample_data["query"].shape
        assert key_rotated.shape == sample_data["key"].shape

        # Verify outputs are different from inputs
        assert not torch.allclose(query_rotated, sample_data["query"])
        assert not torch.allclose(key_rotated, sample_data["key"])

    @pytest.mark.parametrize("share_heads", [True, False], ids=["shared", "not_shared"])
    def test_share_heads(
        self, share_heads: bool, base_config: dict[str, Any], device: str
    ):
        """Test initialization with head sharing enabled and disabled."""
        n_heads = base_config["n_heads"]
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=n_heads,
            share_heads=share_heads,
            dtype=base_config["dtype"],
        ).to(device)

        assert rope.share_heads == share_heads

        expected_n_heads = 1 if share_heads else n_heads
        assert rope.freqs[0].shape[1] == expected_n_heads

        # Create query with same embedding for each head
        head_dim = base_config["embed_dim"] // n_heads
        query = torch.randn(1, 1, 1, head_dim, device=device)
        query = query.expand(-1, -1, n_heads, -1).reshape(1, 1, -1)
        for i in range(n_heads):
            assert torch.equal(
                query[..., :head_dim], query[..., i * head_dim : (i + 1) * head_dim]
            )

        query_pos = torch.ones(1, 1, base_config["position_dim"], device=device) * 5

        # Process the query
        query_rotated = rope(query, query_pos)

        # Reshape to separate heads
        query_rotated_heads = query_rotated.reshape(1, 1, n_heads, head_dim)

        # Check if all heads have the same rotation if heads shared
        for i in range(1, n_heads):
            allclose = torch.allclose(
                query_rotated_heads[0, 0, 0], query_rotated_heads[0, 0, i]
            )
            if share_heads:
                assert allclose
            else:
                assert not allclose

    def test_forward_key_without_pos(
        self, base_config: dict[str, Any], sample_data, device: str
    ):
        """Test forward pass with key but no key positions (should use query positions)."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        # Call without key_pos
        query_rotated, key_rotated_1 = rope(
            sample_data["query"], sample_data["query_pos"], sample_data["key"]
        )

        # Call with key_pos = query_pos explicitly
        query_rotated_2, key_rotated_2 = rope(
            sample_data["query"],
            sample_data["query_pos"],
            sample_data["key"],
            sample_data["query_pos"],
        )

        # Results should be identical
        assert torch.allclose(key_rotated_1, key_rotated_2)

    def test_warning_for_normalized_positions(
        self, base_config: dict[str, Any], device: str
    ):
        """Test warning is raised for potentially normalized coordinates."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        batch_size, seq_len = 2, 10
        query = torch.randn(
            batch_size,
            seq_len,
            base_config["embed_dim"],
            dtype=base_config["dtype"],
            device=device,
        )

        # Create positions in [0, 1] range (normalized)
        query_pos = torch.rand(
            batch_size,
            seq_len,
            base_config["position_dim"],
            dtype=base_config["dtype"],
            device=device,
        )

        with pytest.warns(UserWarning, match="potentially normalized coordinates"):
            rope(query, query_pos)

    @pytest.mark.parametrize(
        "batch_shape",
        [
            (2,),  # Batch dim only
            (2, 10),  # Batch and sequence dims
            (2, 3, 4),  # Batch, width, height
            (2, 3, 4, 5),  # 4D batch shape
        ],
    )
    def test_different_batch_shapes(
        self, batch_shape, base_config: dict[str, Any], device: str
    ):
        """Test forward pass with different batch shapes."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        # Create query and positions with the specified batch shape
        query = torch.randn(
            *batch_shape,
            base_config["embed_dim"],
            dtype=base_config["dtype"],
            device=device,
        )
        query_pos = (
            torch.rand(
                *batch_shape,
                base_config["position_dim"],
                dtype=base_config["dtype"],
                device=device,
            )
            * 10
        )

        # Process
        query_rotated = rope(query, query_pos)

        # Verify shape is preserved
        assert query_rotated.shape == query.shape
