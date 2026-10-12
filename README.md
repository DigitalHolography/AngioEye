# AngioEye

AngioEye runs statistical tools over batches of EyeFlow HDF5 exports. EyeFlow
calculates the waveform and model metrics; AngioEye aggregates those exported
results, compares cohorts, and produces tables, figures, scores, and HTML reports.

The former Postprocess tools are now **Pipelines**. The application has one
Pipelines library and runs each selected tool once over the batch.

## Installation

Python 3.10 or later is required. From a checkout:

```powershell
uv sync
uv run angioeye
```

Or create a virtual environment and install with pip:

```powershell
python -m venv .venv
.venv/Scripts/activate
pip install -e .
angioeye
```

All statistical and GUI dependencies are included in the basic installation.
For development, use `uv sync --extra dev` or `pip install -e ".[dev]"`.

## Inputs and processing branches

Select one EyeFlow `.h5`/`.hdf5` file, multiple HDF5 files, one folder scanned
recursively, or one ZIP containing HDF5 files. AppleDouble and `__MACOSX` metadata
files are ignored. Folder names are preserved as cohort labels.

Select the EyeFlow processing branch to analyze:

- `Processing` reads `/Processing/Metrics`.
- `ProcessingAlt` reads `/ProcessingAlt/Metrics`.

The supplied `260312_GUJ_4_EF.h5` example contains both branches, with
`waveform_shape_metrics`, `absolute_waveform_metrics`, and
`lowrank_waveform_decomposition` exports. A missing selected branch is reported
as an input error. Every tool checks the exported metric groups it requires;
AngioEye does not run metric calculations to fill missing groups.

## Running statistical pipelines

In the GUI, choose inputs and an output folder in **Run**, choose the tools in
**Pipelines**, then run the batch. HTML summary and Pipeline Metrics Manifest
are selected initially. Selection preferences are saved between launches;
existing Postprocess preferences migrate to the Pipelines library.

The CLI uses the same batch workflow:

```powershell
angioeye --list-pipelines
angioeye --data data/ --pipelines "HTML summary" "Pipeline Metrics Manifest" --output results/
angioeye-cli --data sample_EF.h5 --pipelines "Fixed Threshold Patient Evaluation" --output results/
angioeye-cli --data cohort.zip --pipelines "Variability and heterogeneity" --processing-branch ProcessingAlt --output results/ --zip
```

`--data` accepts multiple explicit HDF5 paths. `--pipelines` also accepts a text
file containing one tool name per line, or a quoted list literal. Use
`--zip-name statistics.zip` to choose the optional output archive name.
The CLI returns a nonzero status when an input, pipeline, or archive fails.

Available tools include per-file HTML summaries, a metric manifest,
fixed-threshold evaluation, waveform group comparisons, variability and
heterogeneity, composite scoring, and low-rank cohort figures. Windkessel QC
is available for exports containing `Windkessel_RC` metrics.

Group comparisons and composite calibration use cohort subfolders, for example:

```text
cohort/
  control/
    subject_1_EF.h5
  pathology/
    subject_2_EF.h5
```

Composite Scoring requires a recognizable control group and a second group.
Low-rank cohort figures require consecutive numbered group folders, such as
`1_ctrl/`, `2_path/` or `1_baseline/`, `2_flicker/`, `3_recovery/`.
These tools report unsuitable layouts instead of generating empty analyses.

## Results and source preservation

EyeFlow inputs are opened read-only. AngioEye makes independent compact analysis
files under `results/h5/`, preserving group folders and naming files `_AE.h5`.
Each contains the selected branch's metrics under `/Processing/Metrics`, global
beat signals, cardiac timing, and lightweight metadata. Source image stacks,
maps, and segment waveform stacks are omitted.

Derived statistics are stored under `/AngioEye/Statistics` in these analysis
files. Reports go into the results folder, including `html/`, `png/`,
`cohort-results/`, or a tool's report folder. `batch_run.json` records source
paths, the branch, tool results, failures, and the batch summary. Optional ZIP
export includes the results and reports. Use a separate output folder for each
analysis or processing branch.

## Adding a pipeline

Create a module in `src/pipelines/` with a registered batch class or function:

```python
from pipelines import BatchPipeline, PipelineContext, PipelineResult, registerPipeline


@registerPipeline(
    name="My Cohort Summary",
    description="Summarize exported EyeFlow metrics across a batch.",
    required_metrics=["waveform_shape_metrics"],
)
class MyCohortSummary(BatchPipeline):
    def run(self, context: PipelineContext) -> PipelineResult:
        # context.processed_files contains independent analysis copies.
        # context.input_h5_paths contains source paths for provenance.
        report = context.output_dir / "my_summary.txt"
        report.write_text(f"Files: {len(context.processed_files)}", encoding="utf-8")
        return PipelineResult(
            summary="Generated my_summary.txt.",
            generated_paths=[str(report)],
        )
```

Declare Python dependencies with `required_deps`. Use
`required_metric_options=[["metric_a", "metric_a_alternative"], ["metric_b"]]`
for alternative metric groups: choices within a list are OR alternatives,
and separate lists are required together. Tools can constrain input methods
and minimum file counts. Python dependencies determine catalog availability;
metric requirements are checked against the selected EyeFlow inputs at run time.

Installed builds expose editable `pipelines/` modules next to `AngioEye.exe`.
The Pipelines tab provides Open folder and Reload buttons. A checkout launched
through `angioeye` uses the local `src/` tree.

## Development and Windows installer

```powershell
uv run python -m unittest discover -s test
uv run ruff check .
uv run ruff format --check .
.\build_installer.ps1
```

The installer script requires PyInstaller and Inno Setup 6. The default executable
is a GUI build; pass `-Console` to build a console-enabled executable for CLI use.
