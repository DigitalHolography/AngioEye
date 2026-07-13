from __future__ import annotations

from workflows import (
    WorkflowInputError,
    WorkflowWorkSelection,
    resolve_work_selection,
)

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
        try:
            return resolve_work_selection(
                pipeline_names,
                self.app.pipeline_registry,
                selected_postprocess_names,
                self.app.postprocess_registry,
            )
        except WorkflowInputError as exc:
            dialog = services_for(self.app).dialogs
            if exc.title == "No work selected":
                dialog.showwarning(exc.title, exc.message)
            else:
                dialog.showerror(exc.title, exc.message)
            return None
