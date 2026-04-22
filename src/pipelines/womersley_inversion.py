import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline


@registerPipeline(name="womersley inversion")
class WomersleyInversion(ProcessPipeline):
    """
    Placeholder skeleton for the future Womersley inversion pipeline.

    The exact input and output contract is still under definition, so this
    module intentionally keeps only a minimal interface shape for now.
    """

    description = "Placeholder Womersley inversion pipeline."

    # ----------------------------
    # Inputs
    # ----------------------------
    input_velocity_path = "/TODO/input/velocity"
    input_period_path = "/TODO/input/period"

    # ----------------------------
    # Outputs
    # ----------------------------
    output_root = "womersley_inversion"

    def run(self, h5file) -> ProcessResult:
        """
        Temporary no-op entrypoint.

        This keeps the pipeline interface alive while the real input/output
        schema is being defined.
        """
        _ = h5file

        metrics = {
            f"{self.output_root}/status": np.asarray(
                "pending interface definition", dtype="S"
            )
        }

        return ProcessResult(metrics=metrics)
