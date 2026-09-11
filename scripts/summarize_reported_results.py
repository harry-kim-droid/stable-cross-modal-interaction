"""Derive human-readable deltas from the paper-reported CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "paper_reported_results.csv"
OUTPUT = ROOT / "results" / "derived_summary.json"


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


with SOURCE.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))


def find(dataset: str, split: str, method: str) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row["dataset"] == dataset and row["split"] == split and row["method"] == method
    )


ego_base = find("EgoTaskQA", "Overall", "DE-0")
ego_gated = find("EgoTaskQA", "Overall", "FIB-6 Gated")
msr_base = find("MSR-VTT", "Binary", "DE-0")
msr_gated = find("MSR-VTT", "Binary", "FIB-6 Gated")

ego_drop_reduction = as_float(ego_base, "drop_pp") - as_float(ego_gated, "drop_pp")
summary = {
    "provenance": "arithmetic derived from paper-reported Tables 2, 5, and 6",
    "ego_task_qa": {
        "clean_accuracy_gain_pp": round(
            as_float(ego_gated, "clean_accuracy") - as_float(ego_base, "clean_accuracy"), 2
        ),
        "frame_drop_reduction_pp": round(ego_drop_reduction, 2),
        "frame_drop_reduction_percent": round(
            100.0 * ego_drop_reduction / as_float(ego_base, "drop_pp"), 1
        ),
    },
    "msr_vtt": {
        "clean_accuracy_gain_pp": round(
            as_float(msr_gated, "clean_accuracy") - as_float(msr_base, "clean_accuracy"), 2
        ),
        "macro_f1_gain": round(
            as_float(msr_gated, "macro_f1") - as_float(msr_base, "macro_f1"), 3
        ),
        "gated_retention_percent": round(
            100.0
            * as_float(msr_gated, "perturbed_accuracy")
            / as_float(msr_gated, "clean_accuracy"),
            1,
        ),
    },
    "efficiency": {
        "parameter_increase_percent": round(
            100.0
            * (as_float(msr_gated, "params_m") - as_float(msr_base, "params_m"))
            / as_float(msr_base, "params_m"),
            1,
        ),
        "gflops_increase_percent": round(
            100.0
            * (as_float(msr_gated, "gflops") - as_float(msr_base, "gflops"))
            / as_float(msr_base, "gflops"),
            1,
        ),
        "latency_increase_percent": round(
            100.0
            * (as_float(msr_gated, "latency_ms") - as_float(msr_base, "latency_ms"))
            / as_float(msr_base, "latency_ms"),
            1,
        ),
    },
}

OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))

