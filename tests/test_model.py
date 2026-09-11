import torch

from rcmi import BidirectionalGatedFusion, GatedCrossModalUpdate


def test_gate_initializes_to_paper_value() -> None:
    layer = GatedCrossModalUpdate(hidden_dim=16, num_heads=4, gate_init=0.1)
    assert torch.isclose(layer.gamma, torch.tensor(0.1), atol=1e-6)


def test_bounded_update_property() -> None:
    torch.manual_seed(3)
    layer = GatedCrossModalUpdate(hidden_dim=16, num_heads=4, gate_init=0.1)
    target = torch.randn(3, 7, 16)
    context = torch.randn(3, 5, 16)
    updated, diagnostics = layer(target, context)

    assert updated.shape == target.shape
    assert diagnostics.max_bound_violation <= 1e-6
    assert torch.all(diagnostics.update_norm <= diagnostics.bound + 1e-6)


def test_bidirectional_fusion_preserves_stream_shapes() -> None:
    torch.manual_seed(5)
    fusion = BidirectionalGatedFusion(hidden_dim=24, num_heads=4, gate_init=0.1)
    video = torch.randn(2, 16, 24)
    text = torch.randn(2, 9, 24)

    fused_video, fused_text, diagnostics = fusion(video, text)

    assert fused_video.shape == video.shape
    assert fused_text.shape == text.shape
    assert set(diagnostics) == {"video", "text"}
    assert diagnostics["video"].attention.shape[-2:] == (16, 9)
    assert diagnostics["text"].attention.shape[-2:] == (9, 16)


def test_extreme_gate_ablation() -> None:
    torch.manual_seed(9)
    layer = GatedCrossModalUpdate(hidden_dim=8, num_heads=2, gate_init=0.1)
    target = torch.randn(1, 4, 8)
    context = torch.randn(1, 3, 8)

    layer.set_gate(0.0)
    near_identity, _ = layer(target, context)
    assert torch.allclose(near_identity, target, atol=1e-5)

    layer.set_gate(1.0)
    _, diagnostics = layer(target, context)
    assert diagnostics.gamma > 0.99999

