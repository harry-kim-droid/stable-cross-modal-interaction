"""Diagnostics and perturbations for stability experiments."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def attention_entropy(attention: Tensor, eps: float = 1e-12) -> Tensor:
    """Return entropy over the final attention dimension.

    For model attention shaped ``[batch, heads, target, source]``, the result has
    shape ``[batch, heads, target]``. This is a cross-modal attention diagnostic;
    it should not be conflated with the paper's temporal-attention entropy unless
    the supplied attention axis is explicitly temporal.
    """

    probabilities = attention.clamp_min(eps)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return -(probabilities * probabilities.log()).sum(dim=-1)


def bounded_update_report(
    original: Tensor,
    candidate: Tensor,
    updated: Tensor,
    gamma: Tensor | float,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Check Lemma 1 numerically for a batch of feature vectors."""

    gate = torch.as_tensor(gamma, dtype=original.dtype, device=original.device)
    update_norm = torch.linalg.vector_norm(updated - original, dim=-1)
    bound = gate * torch.linalg.vector_norm(candidate - original, dim=-1)
    violation = update_norm - bound
    return {
        "passed": bool(torch.all(violation <= tolerance).item()),
        "max_violation": float(violation.max().detach().cpu()),
        "mean_update_norm": float(update_norm.mean().detach().cpu()),
        "mean_bound": float(bound.mean().detach().cpu()),
    }


def random_frame_drop(video: Tensor, rate: float, seed: int) -> tuple[Tensor, Tensor]:
    """Zero randomly selected frame tokens and return the boolean drop mask."""

    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")
    generator = torch.Generator(device=video.device).manual_seed(seed)
    mask = torch.rand(video.shape[:2], generator=generator, device=video.device) < rate
    if video.shape[1] > 0:
        mask[:, 0] = False  # preserve one token for stable attention behavior
    perturbed = video.masked_fill(mask.unsqueeze(-1), 0.0)
    return perturbed, mask


def gaussian_noise(video: Tensor, sigma: float, seed: int) -> Tensor:
    """Apply zero-mean Gaussian noise with a deterministic seed."""

    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    generator = torch.Generator(device=video.device).manual_seed(seed)
    noise = torch.randn(
        video.shape,
        generator=generator,
        device=video.device,
        dtype=video.dtype,
    )
    return video + sigma * noise

