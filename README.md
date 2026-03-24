# AngioEye

AngioEye is the cohort-analysis engine for retinal Doppler holography. It browses EyeFlow .h5 outputs, reads per-segment metrics, applies QC, compares models, and aggregates results at eye/cohort level (including artery–vein summaries) to help design biomarkers. It exports clean CSV reports for stats, figures, and clinical models.

---

## Setup

### Prerequisites

- Python 3.10 or higher.
- It is highly recommended to use a virtual environment.

This project uses a `pyproject.toml` to describe all requirements needed. To start using it, **it is better to use a Python virtual environment (venv)**.

```sh
# Creates the venv
python -m venv .venv

# To enter the venv
# If you are using Windows PowerShell, you might need to activate the "Exceution" policy
./.venv/Scripts/activate
```

You can easily exit it with the command

```sh
deactivate
```

### 1. Basic Installation (User)

```sh
pip install -e .

# Installs pipeline-specific dependencies (optional)
pip install -e ".[pipelines]"

# Installs postprocess-specific dependencies such as the graphics dashboard (optional)
pip install -e ".[postprocess]"
```

### 2. Development Setup (Contributor)

```sh
# Install all dependencies including dev tools (ruff, pre-commit, pyinstaller)
pip install -e ".[dev,pipelines,postprocess]"

# Initialize pre-commit hooks (optionnal)
pre-commit install
```

> [!NOTE]
> The pre-commit is really usefull to run automatic checks before pushing code, reducing chances of ugly code being pushed.
>
> If a pre-commit hook fails, it will try to fix all needed files, **so you will need to add them again before recreating the commit**.

> [!TIP]
> You can run the linter easily, once the `dev` dependencies are installed, with the command:
>
> ```sh
> # To only run the checks
> lint-tool
>
> # To let the linter try to fix as much as possible
> lint-tool --fix
> ```

---

## Usage

Launch the main application to process files interactively:

### GUI

The GUI handles batch processing for folders, single .h5/.hdf5 files, or .zip archives and lets you run multiple pipelines at once. Batch outputs preserve the input subfolder layout under the chosen output directory (one combined `.h5` per input file).

You can also select batch-level postprocess steps. These run after the selected pipelines finish and before optional zipping, so any generated dashboards, PNGs, or summaries are included in the final output folder or archive.

Use the Pipeline Library tab to decide which pipelines are shown in the main Batch tab. Visibility preferences are saved per user between app launches, including installed builds.
Use the Postprocess Library tab the same way for postprocess steps.

```sh
# Via the entry point
angioeye

# Or via the script
python src/angio_eye.py
```

When you run `angioeye` from inside the repository checkout, the launcher prefers the local `src/` tree so newly added or edited pipelines are picked up without needing a full reinstall.

### CLI

The CLI is designed for batch processing in headless environments or clusters.

```sh
# Via the entry point
angioeye-cli

# Or via the script
python src/cli.py
```

---

## Pipeline System

Pipelines are the heart of AngioEye. To add a new analysis, create a file in `src/pipelines/` with a class inheriting from `ProcessPipeline`.

To register it to the app, add the decorator `@register_pipeline`. You can define any needed imports inside, as well as some more info.

To see more complete examples, check out `src/pipelines/basic_stats.py` and `src/pipelines/dummy_heavy.py`.

### Simple Pipeline Structure

```python
from pipelines import ProcessPipeline, ProcessResult, registerPipeline

@registerPipeline(
    name="My Analysis",
    description="Calculates a custom clinical metric.",
    required_deps=["torch>=2.2"],
)
class MyAnalysis(ProcessPipeline):
    def run(self, h5file):
        import torch
        # 1. Read data using h5py
        # 2. Perform calculations
        # 3. Return metrics

        metrics={"peak_flow": 12.5}

        # Optional attributes applied to the pipeline group.
        attrs = {
            "pipeline_version": "1.0",
            "author": "StaticExample"
        }

        return ProcessResult(
            metrics=metrics,
            attrs=attrs
        )
```

## Postprocess System

Postprocess steps are discovered from `src/postprocess/` in the same spirit as pipelines, but they run once per batch over the generated pipeline output folder.

Use `@registerPostprocess(...)` to declare:

- optional Python package dependencies with `required_deps`
- required pipeline outputs with `required_pipelines`

### Simple Postprocess Structure

```python
from postprocess.core.base import (
    BatchPostprocess,
    PostprocessContext,
    PostprocessResult,
    registerPostprocess,
)


@registerPostprocess(
    name="My Batch Summary",
    description="Aggregate metrics across the generated batch outputs.",
    required_pipelines=["Basic Stats"],
)
class MyBatchSummary(BatchPostprocess):
    def run(self, context: PostprocessContext) -> PostprocessResult:
        report_path = context.output_dir / "my_batch_summary.json"
        report_path.write_text("{}", encoding="utf-8")

        return PostprocessResult(
            summary="Generated my_batch_summary.json.",
            generated_paths=[str(report_path)],
            metadata={"file_count": len(context.processed_files)},
        )
```

Inside a postprocess, you can:

- read `context.output_dir`
- read `context.processed_files`
- read `context.selected_pipelines`
- read `context.input_path`
- read `context.zip_outputs`
- write extra artifacts into `context.output_dir` before optional zipping
- return a short `summary`, explicit `generated_paths`, and structured `metadata`

The included `Graphics Dashboard` postprocess shows the intended pattern: it consumes the `arterial_waveform_shape_metrics` output and generates a cohort dashboard plus PNG exports after the batch finishes.
`Pipeline Metrics Manifest` is a lighter built-in example that writes a JSON inventory of the generated pipeline metric datasets for the batch.
`Postprocess Tutorial` is the minimal reference example: it writes a single JSON file showing every `PostprocessContext` field and the `PostprocessResult` output format.
