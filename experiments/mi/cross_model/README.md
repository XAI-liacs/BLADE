# Cross-model comparison

`compare.py` collects the results of experiments 1 and 2 for Qwen3-8B,
Qwen3-14B and Gemma-3-12B-it from the experiment result folders (tags
`main`, `prescreen`, `cf` for 8B, and the same with `-qwen3-14b` and
`-gemma-3-12b`). It writes to `results/`:

- `layer_curves.png`: probe scores per layer on relative depth;
- `summary.csv` and `summary.md`: one row per finding, one column per
  model.

```bash
uv run python experiments/mi/cross_model/compare.py
```

Write-up: [`../REPORT_replication.md`](../REPORT_replication.md).
