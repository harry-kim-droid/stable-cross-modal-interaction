"""Verify the paper's bounded update property across deterministic random seeds."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcmi import GatedCrossModalUpdate  # noqa: E402

OUTPUT = ROOT / "results" / "smoke_benchmark.csv"
GATES = (0.05, 0.10, 0.25, 0.50, 1.00)
SEEDS = range(5)

rows: list[dict[str, object]] = []
for gate in GATES:
    update_norms: list[float] = []
    bounds: list[float] = []
    max_violation = float("-inf")
    for seed in SEEDS:
        torch.manual_seed(seed)
        video = torch.randn(8, 16, 32)
        query = torch.randn(8, 12, 32)
        layer = GatedCrossModalUpdate(
            hidden_dim=32,
            num_heads=4,
            gate_init=gate,
            mlp_ratio=2.0,
            dropout=0.0,
        )
        layer.eval()
        with torch.inference_mode():
            _, diagnostics = layer(video, query)
        update_norms.append(float(diagnostics.update_norm.mean()))
        bounds.append(float(diagnostics.bound.mean()))
        max_violation = max(max_violation, diagnostics.max_bound_violation)

    rows.append(
        {
            "gate": f"{gate:.2f}",
            "seeds": len(tuple(SEEDS)),
            "mean_update_norm": f"{sum(update_norms) / len(update_norms):.8f}",
            "mean_bound": f"{sum(bounds) / len(bounds):.8f}",
            "max_bound_violation": f"{max_violation:.10f}",
            "passed": max_violation <= 1e-6,
            "scope": "local mechanism smoke test",
        }
    )

with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("gate  mean_update  mean_bound  max_violation  passed")
for row in rows:
    print(
        f"{row['gate']:>4}  {row['mean_update_norm']:>11}  {row['mean_bound']:>10}  "
        f"{row['max_bound_violation']:>13}  {row['passed']}"
    )
print(f"\nWrote {OUTPUT}")

