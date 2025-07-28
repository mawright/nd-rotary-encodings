import pytest

import torch
from torch import Tensor

from nd_rotary_encodings.functional.autograd import (
    calculate_rope_checkpointed,
    rotate_embeddings_checkpointed,
    apply_rope_checkpointed,
)
from nd_rotary_encodings.functional.forward_only import (
    rotate_embeddings_forward_only,
    apply_rope_forward_only,
)


@pytest.mark.cuda_if_available
class TestCalculateRopeCheckpointed:
    def test_forward(self, device):
        batch_dim = 2
        seq_length = 3
        n_heads = 4
        head_dim = 8

        position_dim = 3
        n_freq_groups = 2

        positions = torch.randn(batch_dim, seq_length, position_dim, device=device)
        rope_freqs = torch.randn(
            position_dim, n_freq_groups, n_heads, head_dim // 2, device=device
        )

        out = calculate_rope_checkpointed(positions, rope_freqs)

        assert out is not None
        assert out.shape == (batch_dim, seq_length, n_heads, head_dim // 2)
        assert out.device == positions.device

    def test_backward(self, device):
        batch_dim = 2
        seq_length = 3
        n_heads = 4
        head_dim = 8

        position_dim = 3
        n_freq_groups = 2

        positions = torch.randn(
            batch_dim, seq_length, position_dim, device=device, requires_grad=True
        )
        rope_freqs = torch.randn(
            position_dim,
            n_freq_groups,
            n_heads,
            head_dim // 2,
            device=device,
            requires_grad=True,
        )

        out = calculate_rope_checkpointed(positions, rope_freqs)

        loss = out.sum()
        loss.backward()

        assert positions.grad is not None
        assert rope_freqs.grad is not None

    def test_gradcheck(self, device):
        batch_dim = 2
        seq_length = 3
        n_heads = 4
        head_dim = 8

        position_dim = 3
        n_freq_groups = 2

        positions = torch.randn(
            batch_dim,
            seq_length,
            position_dim,
            device=device,
            requires_grad=True,
            dtype=torch.double,
        )
        rope_freqs = torch.randn(
            position_dim,
            n_freq_groups,
            n_heads,
            head_dim // 2,
            device=device,
            requires_grad=True,
            dtype=torch.double,
        )

        assert torch.autograd.gradcheck(
            calculate_rope_checkpointed, (positions, rope_freqs)
        )


@pytest.mark.cuda_if_available
class TestRotateEmbeddingsCheckpointed:
    def test_forward(self, device):
        batch_dim = 2
        seq_length = 5
        n_heads = 4
        head_dim = 8

        embeddings = torch.randn(
            batch_dim, seq_length, n_heads, head_dim, device=device
        )
        rope_encoding = torch.randn(
            batch_dim, seq_length, n_heads, head_dim // 2, device=device
        )

        out = rotate_embeddings_checkpointed(embeddings, rope_encoding)

        assert out is not None
        assert out.shape == embeddings.shape
        assert out.device == embeddings.device

    def test_backward(self, device):
        batch_dim = 2
        seq_length = 5
        n_heads = 4
        head_dim = 8

        embeddings = torch.randn(
            batch_dim, seq_length, n_heads, head_dim, device=device, requires_grad=True
        )
        rope_encoding = torch.randn(
            batch_dim,
            seq_length,
            n_heads,
            head_dim // 2,
            device=device,
            requires_grad=True,
        )

        out = rotate_embeddings_checkpointed(embeddings, rope_encoding)

        loss = out.sum()
        loss.backward()

        assert embeddings.grad is not None
        assert rope_encoding.grad is not None

    def test_gradcheck(self, device):
        batch_dim = 2
        seq_length = 5
        n_heads = 4
        head_dim = 8

        embeddings = torch.randn(
            batch_dim,
            seq_length,
            n_heads,
            head_dim,
            device=device,
            requires_grad=True,
            dtype=torch.double,
        )
        rope_encoding = torch.randn(
            batch_dim,
            seq_length,
            n_heads,
            head_dim // 2,
            device=device,
            requires_grad=True,
            dtype=torch.double,
        )

        assert torch.autograd.gradcheck(
            rotate_embeddings_checkpointed, (embeddings, rope_encoding)
        )


@pytest.mark.cuda_if_available
class TestRotateEmbeddingsForwardOnly:
    def test_forward(self, device):
        batch_dim = 2
        seq_length = 5
        n_heads = 4
        head_dim = 8

        embeddings = torch.randn(
            batch_dim, seq_length, n_heads, head_dim, device=device
        )
        rope_encoding = torch.randn(
            batch_dim, seq_length, n_heads, head_dim // 2, device=device
        )

        embeddings_copy = embeddings.clone()

        # Test with inplace=False
        out = rotate_embeddings_forward_only(embeddings, rope_encoding, inplace=False)

        assert not torch.allclose(embeddings, out)
        assert torch.equal(embeddings, embeddings_copy)

        # Test with inplace=True
        out_2 = rotate_embeddings_forward_only(embeddings, rope_encoding, inplace=True)

        assert torch.allclose(out, out_2)
        assert torch.equal(embeddings, out_2)
        assert not torch.allclose(embeddings, embeddings_copy)


@pytest.mark.cuda_if_available
class TestApplyRopeCheckpointed:
    def test_forward(self, device):
        batch_dim = 2
        seq_length = 5
        n_heads = 4
        head_dim = 8

        position_dim = 3
        n_freq_groups = 2

        embeddings = torch.randn(
            batch_dim, seq_length, n_heads, head_dim, device=device
        )
        positions = torch.randn(batch_dim, seq_length, position_dim, device=device)
        rope_freqs = torch.randn(
            position_dim,
            n_freq_groups,
            n_heads,
            head_dim // 2,
            device=device,
        )

        out = apply_rope_checkpointed(embeddings, positions, rope_freqs)

        assert out is not None
        assert isinstance(out, Tensor)
        assert out.shape == embeddings.shape
        assert out.device == embeddings.device
        assert not torch.equal(embeddings, out)

    def test_backward(self, device):
        batch_dim = 2
        seq_length = 5
        n_heads = 4
        head_dim = 8

        position_dim = 3
        n_freq_groups = 2

        embeddings = torch.randn(
            batch_dim, seq_length, n_heads, head_dim, device=device, requires_grad=True
        )
        positions = torch.randn(
            batch_dim, seq_length, position_dim, device=device, requires_grad=True
        )
        rope_freqs = torch.randn(
            position_dim,
            n_freq_groups,
            n_heads,
            head_dim // 2,
            device=device,
            requires_grad=True,
        )

        out = apply_rope_checkpointed(embeddings, positions, rope_freqs)
        assert isinstance(out, Tensor)

        loss = out.sum()
        loss.backward()

        assert embeddings.grad is not None
        assert positions.grad is not None
        assert rope_freqs.grad is not None

    def test_gradcheck(self, device):
        batch_dim = 2
        seq_length = 5
        n_heads = 4
        head_dim = 8

        position_dim = 3
        n_freq_groups = 2

        embeddings = torch.randn(
            batch_dim,
            seq_length,
            n_heads,
            head_dim,
            device=device,
            requires_grad=True,
            dtype=torch.double,
        )
        positions = torch.randn(
            batch_dim,
            seq_length,
            position_dim,
            device=device,
            requires_grad=True,
            dtype=torch.double,
        )
        rope_freqs = torch.randn(
            position_dim,
            n_freq_groups,
            n_heads,
            head_dim // 2,
            device=device,
            requires_grad=True,
            dtype=torch.double,
        )

        assert torch.autograd.gradcheck(
            apply_rope_checkpointed, (embeddings, positions, rope_freqs)
        )

    def test_gradcheck_with_key(self, device):
        batch_dim = 2
        seq_length = 5
        n_heads = 4
        head_dim = 8

        position_dim = 3
        n_freq_groups = 2

        query_embeddings = torch.randn(
            batch_dim,
            seq_length,
            n_heads,
            head_dim,
            device=device,
            requires_grad=True,
            dtype=torch.double,
        )
        positions = torch.randn(
            batch_dim,
            seq_length,
            position_dim,
            device=device,
            requires_grad=True,
            dtype=torch.double,
        )
        rope_freqs = torch.randn(
            position_dim,
            n_freq_groups,
            n_heads,
            head_dim // 2,
            device=device,
            requires_grad=True,
            dtype=torch.double,
        )
        key_embeddings = torch.randn(
            batch_dim,
            seq_length,
            n_heads,
            head_dim,
            device=device,
            requires_grad=True,
            dtype=torch.double,
        )

        assert torch.autograd.gradcheck(
            apply_rope_checkpointed,
            (query_embeddings, positions, rope_freqs, key_embeddings),
        )


@pytest.mark.cuda_if_available
class TestApplyRopeForwardOnly:
    def test_forward(self, device):
        batch_dim = 2
        seq_length = 5
        n_heads = 4
        head_dim = 8

        position_dim = 3
        n_freq_groups = 2

        embeddings = torch.randn(
            batch_dim, seq_length, n_heads, head_dim, device=device
        )
        positions = torch.randn(batch_dim, seq_length, position_dim, device=device)
        rope_freqs = torch.randn(
            position_dim,
            n_freq_groups,
            n_heads,
            head_dim // 2,
            device=device,
        )

        embeddings_copy = embeddings.clone()

        # Test with inplace=False
        out = apply_rope_forward_only(embeddings, positions, rope_freqs, inplace=False)
        assert isinstance(out, Tensor)

        assert not torch.allclose(embeddings, out)
        assert torch.equal(embeddings, embeddings_copy)

        # Test with inplace=True
        out_2 = apply_rope_forward_only(embeddings, positions, rope_freqs, inplace=True)
        assert isinstance(out_2, Tensor)

        assert torch.allclose(out, out_2)
        assert torch.equal(embeddings, out_2)
        assert not torch.allclose(embeddings, embeddings_copy)
