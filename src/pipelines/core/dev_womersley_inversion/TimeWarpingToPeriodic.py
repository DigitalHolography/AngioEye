from __future__ import annotations

import numpy as np

from ..base import ProcessPipeline, ProcessResult, registerPipeline


@registerPipeline(name="womersley inversion")
class WomersleyInversion(ProcessPipeline):
    description = "Placeholder Womersley inversion pipeline."

    input_velocity_path = "/TODO/input/velocity"
    input_period_path = "/TODO/input/period"
    output_root = "womersley_inversion"

    @staticmethod
    def _nextpow2(n: int) -> int:
        n = int(n)
        if n < 1:
            raise ValueError(f"Expected a positive integer, got {n}")
        return 1 << (n - 1).bit_length()

    @staticmethod
    def _cycle_to_mat(curr_cycle):
        return np.concatenate(
            [np.atleast_2d(np.asarray(frame, dtype=float)) for frame in curr_cycle],
            axis=0,
        )

    @classmethod
    def handleSeg(cls, seg, cycleLength, sysIdxList):
        all_bounds = np.asarray(sysIdxList, dtype=int).ravel()
        cycleCount = all_bounds.size - 1
        warped_seg = []

        for i in range(cycleCount):
            start_idx = int(all_bounds[i]) - 1
            end_idx = int(all_bounds[i + 1]) - 1
            if end_idx <= start_idx:
                continue

            curr_cycle = seg[start_idx:end_idx]
            if len(curr_cycle) == 0:
                continue

            mat = cls._cycle_to_mat(curr_cycle)
            baseCycleLength = mat.shape[1]
            if baseCycleLength == 0:
                continue

            x = np.arange(baseCycleLength, dtype=float)
            xq = np.linspace(0.0, baseCycleLength - 1, cycleLength)
            interpolated_mat = np.empty((mat.shape[0], cycleLength), dtype=float)

            for r in range(mat.shape[0]):
                interpolated_mat[r] = np.interp(xq, x, mat[r])

            warped_seg.extend(
                [interpolated_mat[r : r + 1, :].copy() for r in range(mat.shape[0])]
            )

        return warped_seg

    @classmethod
    def TimeWarpingToPeriodic(cls, v_profiles_cell, sys_idx_list):
        all_bounds = np.asarray(sys_idx_list, dtype=int).ravel()
        if all_bounds.size < 2:
            raise ValueError("sys_idx_list must contain at least two boundaries")

        max_base_cycle_length = 0
        for row in v_profiles_cell:
            for seg in row:
                if seg is None or len(seg) == 0:
                    continue

                for k in range(all_bounds.size - 1):
                    start_idx = int(all_bounds[k])
                    end_idx = int(all_bounds[k + 1]) - 1
                    if end_idx <= start_idx:
                        continue

                    curr_cycle = seg[start_idx:end_idx]
                    if len(curr_cycle) == 0:
                        continue

                    mat = cls._cycle_to_mat(curr_cycle)
                    max_base_cycle_length = max(max_base_cycle_length, mat.shape[1])

        cycle_length = cls._nextpow2(max_base_cycle_length)

        for i, row in enumerate(v_profiles_cell):
            for j, seg in enumerate(row):
                if seg is None or len(seg) == 0:
                    continue

                v_profiles_cell[i][j] = cls.handleSeg(seg, cycle_length, all_bounds)

        return v_profiles_cell

    def run(self, h5file) -> ProcessResult:
        _ = h5file
        metrics = {
            f"{self.output_root}/status": np.asarray(
                "pending interface definition", dtype="S"
            )
        }
        return ProcessResult(metrics=metrics)
