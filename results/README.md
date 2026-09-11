# Result provenance

`paper_reported_results.csv` is a structured transcription of Tables 2, 4, 5, and 6 from the paper. It is **reported evidence**, not the output of a rerun in this repository.

`derived_summary.json` is generated from the CSV by `scripts/summarize_reported_results.py`. It contains arithmetic deltas only.

`smoke_benchmark.csv` is generated locally by `scripts/run_smoke_benchmark.py`. It verifies the Bounded Update Property on random feature tensors and is not a VideoQA benchmark.

Keeping these files separate prevents a mechanism-level smoke test from being misrepresented as a reproduction of EgoTaskQA or MSR-VTT.

