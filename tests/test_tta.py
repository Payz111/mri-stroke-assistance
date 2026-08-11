"""Tests for flip-based test-time augmentation.

TTA is enabled by default in scripts/evaluate.py, so it is part of how the
published metrics are produced. These tests use deterministic stand-in models
rather than a trained network, to check the averaging and un-flipping logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.inference.tta import predict_with_tta


class ConstantLogits(nn.Module):
    """Returns the same logit everywhere, ignoring the input."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1, *x.shape[2:]), self.value)


class EchoFirstChannel(nn.Module):
    """Passes the first input channel through as logits.

    This makes the model exactly flip-equivariant, so TTA must be a no-op.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


class LeftHalfDetector(nn.Module):
    """Fires on the left half of the volume regardless of the input.

    Deliberately not flip-equivariant, which is what makes flip-averaging
    observable.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.full((x.shape[0], 1, *x.shape[2:]), -10.0)
        half = x.shape[2] // 2
        out[:, :, :half] = 10.0
        return out


def _image(shape=(1, 3, 8, 6, 4)) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(*shape)


class TestPredictWithTTA:
    def test_output_shape_and_probability_range(self):
        out = predict_with_tta(ConstantLogits(0.0), _image())

        assert out.shape == (1, 1, 8, 6, 4)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

    def test_constant_model_averages_to_the_same_probability(self):
        out = predict_with_tta(ConstantLogits(2.0), _image())

        expected = torch.sigmoid(torch.tensor(2.0))
        assert torch.allclose(out, expected.expand_as(out), atol=1e-6)

    def test_equivariant_model_is_unchanged_by_tta(self):
        """If the model commutes with the flip, averaging must not alter it."""
        image = _image()

        tta_out = predict_with_tta(EchoFirstChannel(), image)
        plain = torch.sigmoid(image[:, :1])

        assert torch.allclose(tta_out, plain, atol=1e-6)

    def test_asymmetric_model_is_averaged_over_the_flip(self):
        """A left-half detector, averaged with its mirror, fires on both halves at 0.5."""
        image = _image(shape=(1, 3, 8, 6, 4))

        out = predict_with_tta(LeftHalfDetector(), image, flip_axes=[2])

        # Both halves end up as the mean of ~1.0 and ~0.0
        assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-4)

    def test_without_flips_the_raw_asymmetry_survives(self):
        """Baseline for the averaging test above: with no flips, the model's own
        left/right asymmetry must come through untouched. If this passes and the
        averaging test also passes, the flipped pass really is mirrored back
        before averaging -- forgetting to un-flip would leave the left half at
        1.0 instead of 0.5."""
        image = _image(shape=(1, 3, 8, 6, 4))

        no_tta = predict_with_tta(LeftHalfDetector(), image, flip_axes=[])

        assert torch.all(no_tta[:, :, :4] > 0.9)
        assert torch.all(no_tta[:, :, 4:] < 0.1)

    def test_multiple_flip_axes_average_over_all_passes(self):
        out = predict_with_tta(ConstantLogits(1.0), _image(), flip_axes=[2, 3, 4])

        expected = torch.sigmoid(torch.tensor(1.0))
        assert torch.allclose(out, expected.expand_as(out), atol=1e-6)

    def test_does_not_require_grad(self):
        """Inference must not build a graph."""
        out = predict_with_tta(ConstantLogits(0.5), _image())

        assert not out.requires_grad
