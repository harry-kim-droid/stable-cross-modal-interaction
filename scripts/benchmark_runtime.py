"""Measure inference latency, throughput, and bounded-update reliability.

The benchmark is intentionally mechanism-level. It uses fixed random embeddings so
the reported runtime reflects the fusion module rather than data loading or model
training. Results are machine-specific and must not be compared as VideoQA quality.
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcmi import BidirectionalGatedFusion  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sorted sample."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--video-tokens", type=int, default=16)
    parser.add_argument("--text-tokens", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output", default=str(ROOT / "results" / "runtime_benchmark.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")
    if args.hidden_dim % args.num_heads:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(7)
    device = torch.device(args.device)
    model = BidirectionalGatedFusion(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        gate_init=0.1,
        mlp_ratio=2.0,
        dropout=0.0,
    ).to(device)
    model.eval()
    video = torch.randn(
        args.batch_size, args.video_tokens, args.hidden_dim, device=device
    )
    text = torch.randn(
        args.batch_size, args.text_tokens, args.hidden_dim, device=device
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(video, text)
        synchronize(device)

        latencies_ms: list[float] = []
        bound_failures = 0
        max_violation = float("-inf")
        for _ in range(args.iterations):
            started = time.perf_counter()
            _, _, diagnostics = model(video, text)
            synchronize(device)
            latencies_ms.append((time.perf_counter() - started) * 1_000.0)

            violations = (
                diagnostics["video"].max_bound_violation,
                diagnostics["text"].max_bound_violation,
            )
            max_violation = max(max_violation, *violations)
            bound_failures += sum(value > 1e-6 for value in violations)

    mean_ms = statistics.fmean(latencies_ms)
    samples_per_second = args.batch_size * 1_000.0 / mean_ms
    tokens_per_iteration = args.batch_size * (
        args.video_tokens + args.text_tokens
    )
    tokens_per_second = tokens_per_iteration * 1_000.0 / mean_ms
    checks = args.iterations * 2
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    row = {
        "scope": "local mechanism inference",
        "device": str(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "batch_size": args.batch_size,
        "video_tokens": args.video_tokens,
        "text_tokens": args.text_tokens,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "parameter_count": parameter_count,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "mean_latency_ms": f"{mean_ms:.4f}",
        "p50_latency_ms": f"{percentile(latencies_ms, 0.50):.4f}",
        "p95_latency_ms": f"{percentile(latencies_ms, 0.95):.4f}",
        "samples_per_second": f"{samples_per_second:.2f}",
        "tokens_per_second": f"{tokens_per_second:.2f}",
        "bound_checks": checks,
        "bound_failures": bound_failures,
        "bound_failure_rate": f"{bound_failures / checks:.6f}",
        "max_bound_violation": f"{max_violation:.10f}",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print(f"mean latency: {row['mean_latency_ms']} ms")
    print(f"p95 latency:  {row['p95_latency_ms']} ms")
    print(f"throughput:   {row['samples_per_second']} samples/s")
    print(f"bound checks: {checks} ({bound_failures} failures)")
    print(f"wrote:        {output}")


if __name__ == "__main__":
    main()
