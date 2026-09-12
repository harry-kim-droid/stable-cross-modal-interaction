# From an ambiguous request to an executable technical specification

This note records how an open-ended portfolio request was translated into a
testable engineering scope. It is intentionally written as a decision record,
not as a claim that the paper authors' full training system was recovered.

## 1. Initial request

The starting request combined several outcomes without an implementation contract:

- make the paper's code runnable on GitHub;
- compare it numerically with prior methods;
- document the architecture and experiments; and
- provide a FastAPI or web demonstration.

The paper PDF described the mechanism and reported benchmark tables, but the local
materials did not contain original training code, checkpoints, datasets, or the
exact training configuration.

## 2. Risks hidden inside the request

| Ambiguity | Engineering risk | Resolution |
|---|---|---|
| “Reproduce the paper” | A mechanism reimplementation could be mislabeled as a full experimental rerun. | Separate paper-reported numbers from locally computed evidence in both filenames and UI copy. |
| “Compare performance” | Accuracy and runtime could be mixed, or results could lack provenance. | Transcribe paper tables with source columns; generate derived deltas separately; label runtime as machine-local. |
| “Runnable” | A notebook-only artifact may not be testable or reusable. | Package the operator as an importable PyTorch module with tests, scripts, API, Dockerfile, and CI. |
| “Demo” | Hosting the full PyTorch service may be unnecessary for explaining the invariant. | Keep the complete FastAPI API locally and publish a server-independent browser calculator for the core bound. |
| Stable interaction | A visually plausible update may still violate the stated mathematical constraint. | Return per-stream diagnostics and assert the bounded update property across gates and random seeds. |

## 3. Acceptance criteria

The first release is complete only when all of the following are true:

1. `GatedCrossModalUpdate` implements the published convex update explicitly.
2. Bidirectional fusion reads the same pre-update video and text state, avoiding
   update-order leakage.
3. Automated tests cover shapes, validation, gate behavior, API contracts, and
   the bounded update invariant.
4. A deterministic smoke benchmark evaluates five gate values over five seeds
   and reports zero violations above `1e-6`.
5. Paper-reported results and local results remain in separate artifacts.
6. A new user can install, test, benchmark, and start the API from documented
   commands.
7. The public demo explains and calculates the core bound without requiring a
   GPU, model checkpoint, or private dataset.

## 4. System boundary

```text
Paper equation and reported tables
              │
              ▼
     PyTorch fusion module ───► invariant diagnostics
              │                         │
              ├──► pytest / smoke test ─┘
              │
              ├──► FastAPI service (local full mechanism demo)
              │
              └──► static public explainer (browser-side bound calculator)
```

### In scope

- the backbone-level gated interaction operator;
- deterministic mechanism validation;
- transparent transcription and arithmetic comparison of published results;
- an embedding-level API contract; and
- deployable documentation and an interactive explainer.

### Explicitly out of scope

- claiming an independent EgoTaskQA or MSR-VTT rerun;
- reconstructing missing training hyperparameters;
- distributing third-party data or checkpoints; and
- presenting random-embedding diagnostics as task accuracy.

## 5. Measurable operating evidence

Two measurement paths answer different questions:

- `scripts/run_smoke_benchmark.py` asks whether the mathematical bound holds
  across gates and seeds.
- `scripts/benchmark_runtime.py` asks how fast the isolated bidirectional fusion
  operator runs on the recorded local environment and whether any bound check
  fails during timed inference.

Every runtime row records tensor shape, device, library version, iteration count,
parameter count, mean/p50/p95 latency, throughput, failure count, and maximum
numerical violation. This makes performance claims reproducible and prevents a
single unlabeled number from becoming a misleading headline.

## 6. Trade-offs and next increment

The browser demo favors universal access and zero server cost; it demonstrates the
invariant but does not execute PyTorch. The FastAPI service exposes the PyTorch
operator and diagnostics but needs a Python runtime. A production increment would
add a hosted inference service, request tracing, load testing, and a dataset-backed
evaluation only after model weights and licensing are resolved.
