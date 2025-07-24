import pytest
import torch

from nd_rotary_encodings.position_encoding_layer.utils import (
    validate_atleast_nd,
    validate_nd,
    can_broadcast_shapes,
)


@pytest.mark.cpu_and_cuda
class TestValidate:
    def test_validate_nd(self, device):
        tensor = torch.randn(4, 5, 6, device=device)
        validate_nd(tensor, 3)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected tensor to be 4D",
        ):
            validate_nd(tensor, 4)

    def test_validate_at_least_nd(self, device):
        tensor = torch.randn(4, 5, 6, device=device)
        validate_atleast_nd(tensor, 3)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected tensor to have at least",
        ):
            validate_atleast_nd(tensor, 4)

class TestCanBroadcastShapes:
    def test_single_shape(self):
        shapes = [[1, 2, 3]]
        assert can_broadcast_shapes(shapes)

    def test_cannot_broadcast(self):
        shapes = [[10, 10], [20, 20]]
        assert not can_broadcast_shapes(shapes)
