# AngioEye

AngioEye is the cohort-analysis engine for retinal Doppler holography. It browses EyeFlow .h5 outputs, reads per-segment metrics, applies QC, compares models, and aggregates results at eye/cohort level (including artery–vein summaries) to help design biomarkers. It exports clean CSV reports for stats, figures, and clinical models.

---

## Setup

### Prerequisites

- Python 3.10 or higher.

### Manual Setup / Dev

This project uses a `pyproject.toml` to describe all requirements needed. To start using it, **it is better to use a Python virtual environment (venv)**.

1. **Create a python virtual environment**

```sh
# Creates the venv
python -m venv .venv

# To enter the venv
./.venv/Scripts/activate
```

You can easily exit it with the command

```sh
deactivate
```

2. **Install the core dependencies**

```sh
pip install -e .
```

3. **Install pipeline-specific dependencies** (optional)

```sh
pip install -e .[pipelines]
```

4. **Install dev-specific dependencies** (optional)

```sh
pip install -e .[dev]
```

> [!TIP]
> You can install all depencies in one go with `pip install -e .[dev,pipelines]`

---

## Usage

Launch the main application to process files interactively:

```sh
# Via the entry point
angioeye

# Or via the script
python src/angio_eye.py
```

A CLI version also exists

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
from pipelines import ProcessPipeline, ProcessResult

class MyAnalysis(ProcessPipeline):
    description = "Calculates a custom clinical metric."

    def run(self, h5file):
        # 1. Read data using h5py
        # 2. Perform calculations
        # 3. Return metrics and artifacts

        metrics={"peak_flow": 12.5}
        artifacts = {"note": "Static data for demonstration"}

        # Optional attributes applied to the pipeline group and the root file.
        attrs = {
            "pipeline_version": "1.0",
            "author": "StaticExample"
        }

        file_attrs = {"example_generated": True}

        return ProcessResult(
            metrics=metrics,
            artifacts=artifacts,
            attrs=attrs,
            file_attrs=file_attrs
        )
```
