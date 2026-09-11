"""FastAPI service for the paper portfolio demo."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rcmi import BidirectionalGatedFusion, attention_entropy  # noqa: E402


class BoundRequest(BaseModel):
    gamma: float = Field(0.1, ge=0.0, le=1.0)
    candidate_distance: float = Field(3.2, ge=0.0, le=1_000_000.0)


class FuseRequest(BaseModel):
    video: list[list[float]]
    query: list[list[float]]
    gamma: float = Field(0.1, ge=0.0, le=1.0)

    @field_validator("video", "query")
    @classmethod
    def validate_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        if not value or not value[0]:
            raise ValueError("matrix must contain at least one token and one feature")
        width = len(value[0])
        if len(value) > 128 or width > 768:
            raise ValueError("demo limit is 128 tokens by 768 features")
        if any(len(row) != width for row in value):
            raise ValueError("all rows must have the same feature width")
        return value


app = FastAPI(
    title="Stable Cross-Modal Interaction",
    version="0.1.0",
    description="Executable diagnostics for the gated update in Equation (4).",
)
STATIC_ROOT = PROJECT_ROOT / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "implementation": "equation-4-reference"}


@app.get("/api/paper-results")
def paper_results() -> dict[str, object]:
    result_path = PROJECT_ROOT / "results" / "paper_reported_results.csv"
    with result_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        "provenance": "paper-reported; not independently rerun in this repository",
        "paper": "https://doi.org/10.3390/math14060996",
        "rows": rows,
    }


@app.post("/api/update-bound")
def update_bound(request: BoundRequest) -> dict[str, float | bool]:
    maximum_update = request.gamma * request.candidate_distance
    return {
        "gamma": request.gamma,
        "candidate_distance": request.candidate_distance,
        "maximum_update_norm": maximum_update,
        "bounded": maximum_update <= request.candidate_distance + 1e-12,
    }


@app.post("/api/fuse")
def fuse(request: FuseRequest) -> dict[str, object]:
    if len(request.video[0]) != len(request.query[0]):
        raise HTTPException(status_code=422, detail="video and query dimensions must match")

    hidden_dim = len(request.video[0])
    torch.manual_seed(7)
    model = BidirectionalGatedFusion(
        hidden_dim=hidden_dim,
        num_heads=1,
        gate_init=request.gamma,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    model.eval()
    video = torch.tensor([request.video], dtype=torch.float32)
    query = torch.tensor([request.query], dtype=torch.float32)

    with torch.inference_mode():
        fused_video, fused_query, diagnostics = model(video, query)

    video_diag = diagnostics["video"]
    text_diag = diagnostics["text"]
    return {
        "scope": "embedding-level mechanism demo; not a trained VideoQA prediction",
        "gamma": round(float(video_diag.gamma), 6),
        "fused_video": fused_video.squeeze(0).tolist(),
        "fused_query": fused_query.squeeze(0).tolist(),
        "diagnostics": {
            "video_mean_update_norm": round(float(video_diag.update_norm.mean()), 6),
            "text_mean_update_norm": round(float(text_diag.update_norm.mean()), 6),
            "video_max_bound_violation": round(video_diag.max_bound_violation, 8),
            "text_max_bound_violation": round(text_diag.max_bound_violation, 8),
            "video_cross_modal_attention_entropy": round(
                float(attention_entropy(video_diag.attention).mean()), 6
            ),
            "text_cross_modal_attention_entropy": round(
                float(attention_entropy(text_diag.attention).mean()), 6
            ),
        },
    }

