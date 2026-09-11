"""PyTorch implementation of the paper's gated cross-modal update.

The implementation keeps the mathematical update explicit instead of hiding it in a
larger backbone. This makes the gate and the Bounded Update Property directly
inspectable and lets the module be inserted into an existing transformer stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import torch
from torch import Tensor, nn


def _gate_to_logit(value: float, eps: float = 1e-6) -> float:
    """Convert a gate value in [0, 1] to a finite sigmoid logit."""

    if not 0.0 <= value <= 1.0:
        raise ValueError("gate_init must be between 0 and 1")
    clipped = min(max(value, eps), 1.0 - eps)
    return log(clipped / (1.0 - clipped))


@dataclass
class StreamDiagnostics:
    """Diagnostics for one direction of a gated cross-modal update."""

    gamma: Tensor
    update_norm: Tensor
    candidate_distance: Tensor
    bound: Tensor
    attention: Tensor

    @property
    def max_bound_violation(self) -> float:
        return float((self.update_norm - self.bound).max().detach().cpu())


class CrossModalResidual(nn.Module):
    """Multi-head cross-attention followed by an MLP residual block.

    ``target`` supplies attention queries and ``context`` supplies keys and values.
    The return value is the cross-modal candidate A(target, context) in Equation (4).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        mlp_dim = int(hidden_dim * mlp_ratio)
        self.target_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        target: Tensor,
        context: Tensor,
        context_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        target_normalized = self.target_norm(target)
        context_normalized = self.context_norm(context)
        attended, weights = self.attention(
            query=target_normalized,
            key=context_normalized,
            value=context_normalized,
            key_padding_mask=context_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        residual = target + self.attention_dropout(attended)
        candidate = residual + self.mlp(self.mlp_norm(residual))
        return candidate, weights


class GatedCrossModalUpdate(nn.Module):
    """One directional gated convex update from Equation (4)."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        gate_init: float = 0.1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.operator = CrossModalResidual(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.alpha = nn.Parameter(torch.tensor(_gate_to_logit(gate_init)))

    @property
    def gamma(self) -> Tensor:
        return torch.sigmoid(self.alpha)

    def set_gate(self, value: float) -> None:
        """Set the gate for diagnostics or ablation while keeping it learnable."""

        with torch.no_grad():
            self.alpha.fill_(_gate_to_logit(value))

    def forward(
        self,
        target: Tensor,
        context: Tensor,
        context_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, StreamDiagnostics]:
        candidate, attention = self.operator(
            target=target,
            context=context,
            context_padding_mask=context_padding_mask,
        )
        gamma = self.gamma.to(dtype=target.dtype, device=target.device)
        updated = (1.0 - gamma) * target + gamma * candidate

        update_norm = torch.linalg.vector_norm(updated - target, dim=-1)
        candidate_distance = torch.linalg.vector_norm(candidate - target, dim=-1)
        bound = gamma * candidate_distance
        diagnostics = StreamDiagnostics(
            gamma=gamma,
            update_norm=update_norm,
            candidate_distance=candidate_distance,
            bound=bound,
            attention=attention,
        )
        return updated, diagnostics


class BidirectionalGatedFusion(nn.Module):
    """Apply analogous gated updates to video and language streams."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        gate_init: float = 0.1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.video_update = GatedCrossModalUpdate(
            hidden_dim, num_heads, gate_init, mlp_ratio, dropout
        )
        self.text_update = GatedCrossModalUpdate(
            hidden_dim, num_heads, gate_init, mlp_ratio, dropout
        )

    def set_gate(self, value: float) -> None:
        self.video_update.set_gate(value)
        self.text_update.set_gate(value)

    def forward(
        self,
        video: Tensor,
        text: Tensor,
        video_padding_mask: Tensor | None = None,
        text_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, StreamDiagnostics]]:
        # Both directions read the original inputs, preventing update-order leakage.
        updated_video, video_diagnostics = self.video_update(
            target=video,
            context=text,
            context_padding_mask=text_padding_mask,
        )
        updated_text, text_diagnostics = self.text_update(
            target=text,
            context=video,
            context_padding_mask=video_padding_mask,
        )
        return updated_video, updated_text, {
            "video": video_diagnostics,
            "text": text_diagnostics,
        }

