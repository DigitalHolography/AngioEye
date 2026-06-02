from __future__ import annotations

from postprocess import PostprocessDescriptor
from workflows import WorkflowWorkSelection

from ..services import services_for
from .base import ViewController


class WorkflowSelectionController(ViewController):
    def collect_selection(self) -> WorkflowWorkSelection | None:
        pipeline_names = [
            pipeline.name
            for pipeline in self.app.pipeline_rows
            if pipeline.available
            and self.app.pipeline_visibility.get(pipeline.name, False)
        ]
        selected_postprocess_names = [
            postprocess.name
            for postprocess in self.app.postprocess_rows
            if postprocess.available
            and self.app.postprocess_visibility.get(postprocess.name, False)
        ]
        if not pipeline_names and not selected_postprocess_names:
            services_for(self.app).dialogs.showwarning(
                "No work selected",
                "Select at least one pipeline or postprocess step.",
            )
            return None

        pipelines = []
        missing: list[str] = []
        for name in pipeline_names:
            pipeline = self.app.pipeline_registry.get(name)
            if pipeline is None:
                missing.append(name)
            else:
                pipelines.append(pipeline)
        if missing:
            services_for(self.app).dialogs.showerror(
                "Pipeline missing",
                f"Pipeline(s) not registered: {', '.join(missing)}",
            )
            return None

        postprocesses: list[PostprocessDescriptor] = []
        missing_postprocesses: list[str] = []
        for name in selected_postprocess_names:
            postprocess = self.app.postprocess_registry.get(name)
            if postprocess is None:
                missing_postprocesses.append(name)
            else:
                postprocesses.append(postprocess)
        if missing_postprocesses:
            services_for(self.app).dialogs.showerror(
                "Postprocess missing",
                f"Postprocess step(s) not registered: {', '.join(missing_postprocesses)}",
            )
            return None

        return WorkflowWorkSelection(
            pipeline_names=tuple(pipeline_names),
            pipelines=tuple(pipelines),
            postprocesses=tuple(postprocesses),
        )
