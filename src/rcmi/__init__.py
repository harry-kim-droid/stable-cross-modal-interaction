"""Regularized cross-modal interaction modules."""

from .diagnostics import attention_entropy, bounded_update_report
from .model import BidirectionalGatedFusion, GatedCrossModalUpdate, StreamDiagnostics

__all__ = [
    "BidirectionalGatedFusion",
    "GatedCrossModalUpdate",
    "StreamDiagnostics",
    "attention_entropy",
    "bounded_update_report",
]

__version__ = "0.1.0"

