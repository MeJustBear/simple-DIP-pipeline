# Usage examples

Runnable examples of the `dippipe` package on the sample captures in
`examples/`. These scripts are documentation only and are **not** part of the
installed wheel.

Install the package first (`pip install .` or `pdm install`), then run any
example from the repository root:

```bash
python src/usage/01_full_pipeline.py
python src/usage/02_individual_stages.py
python src/usage/03_universal_raw.py
python src/usage/04_filters_comparison.py
```

| Script | Shows |
|--------|-------|
| `01_full_pipeline.py` | Full pipeline via `build_default_pipeline` + `PipelineConfig`. |
| `02_individual_stages.py` | Running each `Stage` independently, saving/loading artifacts. |
| `03_universal_raw.py` | Universal RAW reading: explicit vs. inferred geometry, any pattern. |
| `04_filters_comparison.py` | Optimized vs. naive Gaussian/bilateral (timing + numerical agreement). |

Outputs are written to `usage_output/` at the repository root.

`_common.py` holds shared paths/constants and is imported by the scripts (run
them from a shell so the script directory is on `sys.path`).
