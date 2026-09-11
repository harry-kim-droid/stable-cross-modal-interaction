# Full benchmark reproduction guide

This repository has two reproducibility levels. They must remain clearly separated.

## Level 1: immediately runnable

The PyTorch module, unit tests, FastAPI demo, result parser, and bounded-update smoke benchmark run without downloading video datasets:

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/run_smoke_benchmark.py
python scripts/summarize_reported_results.py
python -m uvicorn app.main:app --reload
```

This level validates implementation behavior, tensor shapes, gate initialization, bidirectional updates, and Lemma 1. It does not produce EgoTaskQA or MSR-VTT accuracy.

## Level 2: full EgoTaskQA / MSR-VTT experiment

The paper states that all variants use the EgoVLPv2 backbone, the same pretrained weights, identical splits, 20 epochs, and a peak learning rate of `1e-4`. Completing a faithful rerun additionally requires:

1. Authorized access to the EgoTaskQA/Ego4D data.
2. The MSR-VTT videos and the paper's 1:1 binary caption-pair construction.
3. The exact pretrained EgoVLPv2 checkpoint used by the authors.
4. The complete optimizer, scheduler, augmentation, seed, and checkpoint-selection configuration.
5. Four NVIDIA TITAN RTX GPUs or a documented hardware-adjusted training recipe.

The upstream code is available from the archived [facebookresearch/EgoVLPv2](https://github.com/facebookresearch/EgoVLPv2) repository. Its MIT license and archived state should be preserved in any derived integration.

## Backbone integration point

Insert `BidirectionalGatedFusion` at each selected fusion-in-backbone layer. The paper's primary configuration uses six fusion blocks and initializes each gate at `0.1`.

```python
from rcmi import BidirectionalGatedFusion

fusion = BidirectionalGatedFusion(
    hidden_dim=768,
    num_heads=12,
    gate_init=0.1,
)

video_hidden, text_hidden, diagnostics = fusion(
    video_hidden,
    text_hidden,
    video_padding_mask=video_padding_mask,
    text_padding_mask=text_padding_mask,
)
```

For a matched ablation:

- `DE-0`: bypass all fusion modules; preserve the paper's video-only classification head.
- `FIB-6 (Ungated)`: enable six modules and set `gamma=1.0`.
- `FIB-6 (Gated)`: enable six modules with learnable gates initialized at `0.1`.
- Keep data splits, pretrained weights, optimizer, schedule, epoch budget, and checkpoint selection identical.

## Evaluation contract

Report at minimum:

- clean accuracy;
- accuracy after 50% random frame drop;
- accuracy drop in percentage points;
- Macro-F1 to identify majority-class collapse;
- temporal-attention entropy when the relevant temporal weights are available;
- parameter count, GFLOPs, batch latency, and throughput;
- mean and standard deviation over five seeds for interaction-depth ablations.

Do not treat a near-zero accuracy drop as robustness without checking absolute accuracy, Macro-F1, and prediction concentration. The paper explicitly identifies this failure mode as trivial stability.

## Result naming

- Store copied publication values only in `results/paper_reported_results.csv`.
- Store independent reruns under `results/reproduced/` with the commit, checkpoint hash, data split hash, hardware, and seed in the filename or metadata.
- Never overwrite paper-reported data with a local run.

