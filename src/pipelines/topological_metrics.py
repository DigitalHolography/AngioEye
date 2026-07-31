from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from input_output.eyeflow_schema import (
    BEAT_PERIOD,
    SEGMENT_VELOCITY_PER_BEAT,
    get_object,
    missing_paths,
    require_dataset,
)
from math_utils import nanmedian

from .core.base import ProcessResult, with_attrs
from .waveform_shape_metrics import ArterialSegExample

REGION_NAMES = (
    "north_west",
    "north_east",
    "south_west",
    "south_east",
    "north",
    "south",
    "west",
    "east",
)

# The four quadrant names above are kept in the historical output order. For
# ties, use the usual mathematical (counter-clockwise) order starting in the
# north-east: QI, QII, QIII, QIV.
TRIGONOMETRIC_QUADRANT_ORDER = (
    "north_east",
    "north_west",
    "south_west",
    "south_east",
)
TRIGONOMETRIC_QUADRANT_INDICES = tuple(
    REGION_NAMES.index(name) for name in TRIGONOMETRIC_QUADRANT_ORDER
)

OPTIC_DISC_LABEL = -1
REGION_AXIS_LABEL = -2

OPTIC_DISC_CENTER_PATH = "/Segmentation/OpticDisc/CenterXY/value"
OPTIC_DISC_MASK_PATH = "/Segmentation/OpticDisc/Mask/value"
BEAT_PERIOD_PATH = BEAT_PERIOD
EYEFLOW_SPATIAL_Y_INVERTED = True


@dataclass(frozen=True)
class VesselPaths:
    name: str
    branch_ids: str
    branch_label_map: str
    branch_label_map_axes: tuple[str, str]
    branch_label_map_y_inverted: bool
    segment_center_xy: str
    raw_waveform: str
    bandlimited_waveform: str

    def source_metric_path(self, signal_type: str, metric_name: str) -> str:
        segment_group = {
            "raw": "raw_segment",
            "bandlimited": "bandlimited_segment",
        }[signal_type]
        return (
            f"/Processing/Metrics/waveform_shape_metrics/{self.name}/by_segment/"
            f"{segment_group}/{metric_name}"
        )


VESSEL_PATHS = (
    VesselPaths(
        name="artery",
        branch_ids="/Segmentation/Artery/BranchIds/value",
        branch_label_map="/Segmentation/Artery/BranchLabelMap/value",
        branch_label_map_axes=("y", "x"),
        branch_label_map_y_inverted=EYEFLOW_SPATIAL_Y_INVERTED,
        segment_center_xy="/Segmentation/Artery/SegmentCenterXY/value",
        raw_waveform=SEGMENT_VELOCITY_PER_BEAT[("artery", "raw")],
        bandlimited_waveform=SEGMENT_VELOCITY_PER_BEAT[("artery", "bandlimited")],
    ),
    VesselPaths(
        name="vein",
        branch_ids="/Segmentation/Vein/BranchIds/value",
        branch_label_map="/Segmentation/Vein/BranchLabelMap/value",
        branch_label_map_axes=("y", "x"),
        branch_label_map_y_inverted=EYEFLOW_SPATIAL_Y_INVERTED,
        segment_center_xy="/Segmentation/Vein/SegmentCenterXY/value",
        raw_waveform=SEGMENT_VELOCITY_PER_BEAT[("vein", "raw")],
        bandlimited_waveform=SEGMENT_VELOCITY_PER_BEAT[("vein", "bandlimited")],
    ),
)

TOPOLOGICAL_METRICS_REQUIRED_PATHS = (
    OPTIC_DISC_CENTER_PATH,
    OPTIC_DISC_MASK_PATH,
    BEAT_PERIOD_PATH,
    *(
        path
        for vessel in VESSEL_PATHS
        for path in (
            vessel.branch_ids,
            vessel.branch_label_map,
            vessel.segment_center_xy,
            vessel.raw_waveform,
            vessel.bandlimited_waveform,
        )
    ),
)


@dataclass(frozen=True)
class VesselData:
    paths: VesselPaths
    branch_ids: np.ndarray
    branch_label_map: np.ndarray
    segment_center_xy: np.ndarray
    raw_waveform: np.ndarray
    bandlimited_waveform: np.ndarray


class TopologicalMetricsPipeline(ArterialSegExample):
    """Legacy topology computation retained for compatibility, not registered.

    Topological metrics are no longer exposed as an AngioEye pipeline. The
    HTML summary still understands their historical output and renders
    EyeFlow topology maps directly; this class remains importable for old
    callers and fixtures.
    """

    name = "topological_metrics"
    description = (
        "Waveform-shape metrics by optic-disc-centred quadrants, half-planes, "
        "and the individual branches present in each region "
        "(artery + vein; raw + bandlimited)."
    )
    h5_source_label = "EF"
    required_h5_paths = TOPOLOGICAL_METRICS_REQUIRED_PATHS

    def run(self, h5file: h5py.File) -> ProcessResult:
        self._require_inputs(h5file)
        optic_disc_center = self._optic_disc_center(h5file)
        optic_disc_mask = np.asarray(
            require_dataset(h5file, OPTIC_DISC_MASK_PATH),
            dtype=bool,
        )
        if optic_disc_mask.ndim != 2:
            raise ValueError(
                f"{OPTIC_DISC_MASK_PATH} must have shape (y, x), got "
                f"{optic_disc_mask.shape}."
            )
        optic_disc_center, optic_disc_mask = self._normalize_spatial_frame(
            optic_disc_center,
            optic_disc_mask,
        )

        vessel_data = [self._load_vessel(h5file, paths) for paths in VESSEL_PATHS]
        self._validate_spatial_frame(optic_disc_center, optic_disc_mask, vessel_data)

        n_beats = self._shared_beat_count(vessel_data)
        beat_periods = self._beat_periods(h5file, n_beats)
        metric_specs = {item[0]: item for item in self._metric_keys()}
        metrics: dict[str, object] = {}

        metrics["topology/names"] = list(REGION_NAMES)
        metrics["topology/optic_disc/center_xy"] = with_attrs(
            optic_disc_center.astype(np.float32, copy=False),
            {
                "dimDesc": ["coordinate"],
                "coordinate_order": ["x", "y"],
                "coordinate_system": "image_pixel",
                "unit": "pixel",
            },
        )
        # Keep the internal topology frame as [y, x], but serialize image
        # outputs as [x, y] so HDF viewers can use D0=X and D1=Y directly.
        metrics["topology/optic_disc/mask"] = with_attrs(
            optic_disc_mask.T.copy(),
            {
                "dimDesc": ["x", "y"],
                "coordinate_system": "image_pixel",
                "image_origin": "lower_left",
                "y_axis_direction": "increasing_toward_north",
            },
        )

        metric_sources: set[str] = set()
        for vessel in vessel_data:
            membership = self._region_membership(
                vessel.branch_ids,
                vessel.branch_label_map,
                vessel.segment_center_xy,
                optic_disc_center,
            )
            self._pack_topology_outputs(
                metrics,
                vessel,
                optic_disc_mask,
                optic_disc_center,
            )

            for signal_type, waveform in (
                ("raw", vessel.raw_waveform),
                ("bandlimited", vessel.bandlimited_waveform),
            ):
                segment_metrics, source = self._segment_metrics(
                    h5file,
                    vessel,
                    signal_type,
                    waveform,
                    beat_periods,
                    tuple(metric_specs),
                )
                metric_sources.add(source)
                self._pack_region_metrics(
                    metrics,
                    vessel,
                    signal_type,
                    membership,
                    segment_metrics,
                    metric_specs,
                )

        source_summary = ",".join(sorted(metric_sources))
        return ProcessResult(
            metrics=metrics,
            attrs={
                "aggregation": (
                    "median of per-segment metric values over branch-radius "
                    "entries belonging to each region, independently per beat"
                ),
                "boundary_policy": (
                    "x < center_x is west; x >= center_x is east; "
                    "the normalized topology frame uses a bottom-up y-axis: "
                    "y < center_y is south; y >= center_y is north"
                ),
                "branch_aggregation": (
                    "median of all per-segment metric values within an EyeFlow "
                    "branch assigned to one quadrant by BranchLabelMap area, "
                    "independently per beat"
                ),
                "branch_quadrant_assignment": (
                    "largest BranchLabelMap pixel area; ties use trigonometric "
                    "order north-east, north-west, south-west, south-east"
                ),
                "branch_group_naming": "branch_<EyeFlow BranchIds value>",
                "coordinate_order": "x,y",
                "metric_source": source_summary,
                "region_names": list(REGION_NAMES),
                "topology_source": "/Topology",
            },
        )

    @classmethod
    def missing_required_paths(cls, h5file: h5py.File) -> tuple[str, ...]:
        return missing_paths(h5file, cls.required_h5_paths)

    @classmethod
    def _require_inputs(cls, h5file: h5py.File) -> None:
        missing = cls.missing_required_paths(h5file)
        if missing:
            raise KeyError(
                "Missing required EyeFlow topology dataset(s): " + ", ".join(missing)
            )

    @staticmethod
    def _optic_disc_center(h5file: h5py.File) -> np.ndarray:
        center = np.asarray(
            require_dataset(h5file, OPTIC_DISC_CENTER_PATH),
            dtype=float,
        ).reshape(-1)
        if center.size != 2 or not np.all(np.isfinite(center)):
            raise ValueError(
                f"{OPTIC_DISC_CENTER_PATH} must contain one finite (x, y) pair."
            )
        return center

    @staticmethod
    def _normalize_spatial_frame(
        optic_disc_center: np.ndarray,
        optic_disc_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize EyeFlow's inverted-Y spatial frame for topology use."""
        if not EYEFLOW_SPATIAL_Y_INVERTED:
            return optic_disc_center, optic_disc_mask

        normalized_center = optic_disc_center.astype(float, copy=True)
        normalized_center[1] = optic_disc_mask.shape[0] - 1 - normalized_center[1]
        normalized_mask = np.flip(optic_disc_mask, axis=0).copy()
        return normalized_center, normalized_mask

    @staticmethod
    def _load_vessel(h5file: h5py.File, paths: VesselPaths) -> VesselData:
        branch_ids = np.asarray(
            require_dataset(h5file, paths.branch_ids),
            dtype=np.int32,
        ).reshape(-1)
        raw_waveform = np.asarray(
            require_dataset(h5file, paths.raw_waveform),
            dtype=float,
        )
        bandlimited_waveform = np.asarray(
            require_dataset(h5file, paths.bandlimited_waveform),
            dtype=float,
        )
        if branch_ids.size == 0:
            raw_waveform = TopologicalMetricsPipeline._drop_empty_branch_placeholder(
                raw_waveform,
                paths.name,
                "raw",
            )
            bandlimited_waveform = (
                TopologicalMetricsPipeline._drop_empty_branch_placeholder(
                    bandlimited_waveform,
                    paths.name,
                    "bandlimited",
                )
            )
        branch_label_map = np.asarray(
            require_dataset(h5file, paths.branch_label_map),
            dtype=np.int32,
        )
        if paths.branch_label_map_axes == ("x", "y"):
            # Keep this conversion for older inputs that used [x, y]. Current
            # EyeFlow maps use D0=Y and D1=X, i.e. the canonical [y, x] order.
            branch_label_map = branch_label_map.T.copy()
        if paths.branch_label_map_y_inverted:
            # EyeFlow's Y direction is vertically inverted relative to the
            # topology pipeline's lower-left, bottom-up image frame. Normalize
            # it once at the input boundary for both artery and vein maps.
            branch_label_map = np.flip(branch_label_map, axis=0).copy()

        data = VesselData(
            paths=paths,
            branch_ids=branch_ids,
            branch_label_map=branch_label_map,
            segment_center_xy=np.asarray(
                require_dataset(h5file, paths.segment_center_xy),
                dtype=float,
            ),
            raw_waveform=raw_waveform,
            bandlimited_waveform=bandlimited_waveform,
        )
        TopologicalMetricsPipeline._validate_vessel(data)
        return data

    @staticmethod
    def _drop_empty_branch_placeholder(
        waveform: np.ndarray,
        vessel_name: str,
        signal_type: str,
    ) -> np.ndarray:
        if waveform.ndim != 4 or waveform.shape[2] != 1:
            return waveform
        if np.any(np.isfinite(waveform)):
            raise ValueError(
                f"{vessel_name} {signal_type} waveform contains a finite dummy "
                "branch although BranchIds is empty."
            )
        return waveform[:, :, :0, :]

    @staticmethod
    def _validate_vessel(vessel: VesselData) -> None:
        name = vessel.paths.name
        if vessel.branch_label_map.ndim != 2:
            raise ValueError(
                f"{name} BranchLabelMap must have shape (y, x), got "
                f"{vessel.branch_label_map.shape}."
            )
        if vessel.segment_center_xy.ndim != 3 or vessel.segment_center_xy.shape[2] != 2:
            raise ValueError(
                f"{name} SegmentCenterXY must have shape (branch, radius, 2), "
                f"got {vessel.segment_center_xy.shape}."
            )
        if vessel.branch_ids.size != vessel.segment_center_xy.shape[0]:
            raise ValueError(
                f"{name} BranchIds length {vessel.branch_ids.size} does not match "
                f"SegmentCenterXY branch size {vessel.segment_center_xy.shape[0]}."
            )
        if np.unique(vessel.branch_ids).size != vessel.branch_ids.size:
            raise ValueError(f"{name} BranchIds must be unique.")

        known_labels = np.concatenate(
            (np.asarray([0], dtype=np.int32), vessel.branch_ids)
        )
        unexpected_labels = np.setdiff1d(
            np.unique(vessel.branch_label_map),
            known_labels,
        )
        if unexpected_labels.size:
            raise ValueError(
                f"{name} BranchLabelMap contains labels absent from BranchIds: "
                f"{unexpected_labels.tolist()}."
            )
        missing_labels = np.setdiff1d(
            vessel.branch_ids,
            np.unique(vessel.branch_label_map),
        )
        if missing_labels.size:
            raise ValueError(
                f"{name} BranchLabelMap has no pixels for BranchIds: "
                f"{missing_labels.tolist()}."
            )

        expected_tail = vessel.segment_center_xy.shape[:2]
        for signal_type, waveform in (
            ("raw", vessel.raw_waveform),
            ("bandlimited", vessel.bandlimited_waveform),
        ):
            if waveform.ndim != 4:
                raise ValueError(
                    f"{name} {signal_type} waveform must have shape "
                    f"(sample, beat, branch, radius), got {waveform.shape}."
                )
            if waveform.shape[2:] != expected_tail:
                raise ValueError(
                    f"{name} {signal_type} waveform branch-radius shape "
                    f"{waveform.shape[2:]} does not match SegmentCenterXY "
                    f"shape {expected_tail}."
                )

        if vessel.raw_waveform.shape != vessel.bandlimited_waveform.shape:
            raise ValueError(
                f"{name} raw and bandlimited waveform shapes differ: "
                f"{vessel.raw_waveform.shape} versus "
                f"{vessel.bandlimited_waveform.shape}."
            )

    @staticmethod
    def _validate_spatial_frame(
        optic_disc_center: np.ndarray,
        optic_disc_mask: np.ndarray,
        vessels: list[VesselData],
    ) -> None:
        height, width = optic_disc_mask.shape
        center_x, center_y = optic_disc_center
        if not (0 <= center_x < width and 0 <= center_y < height):
            raise ValueError(
                "Optic-disc center lies outside the optic-disc mask image frame."
            )

        for vessel in vessels:
            if vessel.branch_label_map.shape != optic_disc_mask.shape:
                raise ValueError(
                    f"{vessel.paths.name} BranchLabelMap shape "
                    f"{vessel.branch_label_map.shape} does not match optic-disc "
                    f"mask shape {optic_disc_mask.shape}."
                )
            centers = vessel.segment_center_xy
            finite = np.all(np.isfinite(centers), axis=2)
            if not np.any(finite):
                continue
            x = centers[:, :, 0][finite]
            y = centers[:, :, 1][finite]
            if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
                raise ValueError(
                    f"{vessel.paths.name} SegmentCenterXY contains coordinates "
                    "outside the topology image frame."
                )

    @staticmethod
    def _shared_beat_count(vessels: list[VesselData]) -> int:
        counts = {int(vessel.raw_waveform.shape[1]) for vessel in vessels}
        if len(counts) != 1:
            raise ValueError(
                "Artery and vein segment waveforms must have the same beat count."
            )
        return counts.pop()

    @staticmethod
    def _beat_periods(h5file: h5py.File, n_beats: int) -> np.ndarray:
        periods = np.asarray(
            require_dataset(h5file, BEAT_PERIOD_PATH),
            dtype=float,
        ).reshape(-1)
        if periods.size != n_beats:
            raise ValueError(
                f"{BEAT_PERIOD_PATH} contains {periods.size} beat period(s), but "
                f"the segment waveforms contain {n_beats}."
            )
        if np.any(~np.isfinite(periods) | (periods <= 0)):
            raise ValueError("Beat periods must all be finite and positive.")
        return periods.reshape(1, -1)

    @staticmethod
    def _region_membership(
        branch_ids: np.ndarray,
        branch_label_map: np.ndarray,
        segment_center_xy: np.ndarray,
        optic_disc_center: np.ndarray,
    ) -> np.ndarray:
        """Assign every branch and all of its radii to one area-majority region.

        ``BranchLabelMap`` is indexed as ``[y, x]`` while ``SegmentCenterXY``
        stores coordinates as ``[x, y]``. Both are already normalized to the
        pipeline's bottom-up Y frame at the input boundary.
        """
        n_branches, n_radii = segment_center_xy.shape[:2]
        if branch_ids.size != n_branches:
            raise ValueError(
                "BranchIds and SegmentCenterXY must have the same branch count."
            )

        height, width = branch_label_map.shape
        pixel_y, pixel_x = np.indices((height, width), dtype=float)
        west = pixel_x < optic_disc_center[0]
        east = ~west
        south = pixel_y < optic_disc_center[1]
        north = ~south
        quadrant_masks = np.asarray(
            (
                north & west,
                north & east,
                south & west,
                south & east,
            ),
            dtype=bool,
        )

        area_by_quadrant = np.zeros((n_branches, 4), dtype=np.float32)
        for branch_index, branch_id in enumerate(branch_ids):
            branch_pixels = branch_label_map == branch_id
            area_by_quadrant[branch_index] = np.asarray(
                [np.count_nonzero(branch_pixels & mask) for mask in quadrant_masks],
                dtype=np.float32,
            )

        if not np.all(np.any(area_by_quadrant, axis=1)):
            raise ValueError(
                "BranchLabelMap must contain at least one pixel for every branch."
            )

        chosen_quadrants = np.full(n_branches, -1, dtype=np.float32)
        for branch_index, area in enumerate(area_by_quadrant):
            if np.any(area):
                priority_order = np.asarray(TRIGONOMETRIC_QUADRANT_INDICES)
                chosen_quadrants[branch_index] = priority_order[np.argmax(area[priority_order])]

        assigned_quadrants = np.asarray(
            [
                chosen_quadrants == 0,
                chosen_quadrants == 1,
                chosen_quadrants == 2,
                chosen_quadrants == 3,
            ],
            dtype=bool,
        )
        assigned_quadrants = np.broadcast_to(
            assigned_quadrants[:, :, np.newaxis],
            (4, n_branches, n_radii),
        )
        return np.asarray(
            (
                assigned_quadrants[0],
                assigned_quadrants[1],
                assigned_quadrants[2],
                assigned_quadrants[3],
                assigned_quadrants[0] | assigned_quadrants[1],
                assigned_quadrants[2] | assigned_quadrants[3],
                assigned_quadrants[0] | assigned_quadrants[2],
                assigned_quadrants[1] | assigned_quadrants[3],
            ),
            dtype=bool,
        )

    @staticmethod
    def _pack_topology_outputs(
        metrics: dict[str, object],
        vessel: VesselData,
        optic_disc_mask: np.ndarray,
        optic_disc_center: np.ndarray,
    ) -> None:
        prefix = f"topology/{vessel.paths.name}"
        metrics[f"{prefix}/branch_ids"] = with_attrs(
            vessel.branch_ids,
            {"dimDesc": ["branch"]},
        )
        branch_label_map, axis_thickness = (
            TopologicalMetricsPipeline._branch_label_visualization(
                vessel.branch_label_map,
                optic_disc_mask,
                optic_disc_center,
            )
        )
        metrics[f"{prefix}/branch_label_map"] = with_attrs(
            branch_label_map.T.copy(),
            {
                "axis_label": REGION_AXIS_LABEL,
                "axis_thickness_pixels": axis_thickness,
                "dimDesc": ["x", "y"],
                "background_label": 0,
                "branch_labels": "original EyeFlow BranchIds values",
                "coordinate_system": "image_pixel",
                "image_origin": "lower_left",
                "boundary_policy": (
                    "vertical axis at center_x; horizontal axis at center_y"
                ),
                "y_axis_direction": "increasing_toward_north",
                "description": (
                    "Two-dimensional branch label mask with optic-disc and "
                    "region-axis overlays"
                ),
                "optic_disc_label": OPTIC_DISC_LABEL,
                "overlay_priority": "region axes, optic disc, vessel branches",
            },
        )

    @staticmethod
    def _branch_label_visualization(
        branch_label_map: np.ndarray,
        optic_disc_mask: np.ndarray,
        optic_disc_center: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        image = branch_label_map.astype(np.int32, copy=True)
        image[optic_disc_mask] = OPTIC_DISC_LABEL

        height, width = branch_label_map.shape
        axis_thickness = max(3, int(round(min(height, width) / 128.0)))
        center_x = int(np.floor(optic_disc_center[0]))
        center_y = int(np.floor(optic_disc_center[1]))
        x_start = max(0, center_x - axis_thickness // 2)
        x_stop = min(width, x_start + axis_thickness)
        y_start = max(0, center_y - axis_thickness // 2)
        y_stop = min(height, y_start + axis_thickness)
        image[:, x_start:x_stop] = REGION_AXIS_LABEL
        image[y_start:y_stop, :] = REGION_AXIS_LABEL
        return image, axis_thickness

    def _segment_metrics(
        self,
        h5file: h5py.File,
        vessel: VesselData,
        signal_type: str,
        waveform: np.ndarray,
        beat_periods: np.ndarray,
        metric_names: tuple[str, ...],
    ) -> tuple[dict[str, np.ndarray], str]:
        expected_shape = (
            waveform.shape[1],
            waveform.shape[2],
            waveform.shape[3],
        )
        source_metrics: dict[str, np.ndarray] = {}
        for metric_name in metric_names:
            path = vessel.paths.source_metric_path(signal_type, metric_name)
            dataset = get_object(h5file, path)
            if not isinstance(dataset, h5py.Dataset) or dataset.shape != expected_shape:
                break
            source_metrics[metric_name] = np.asarray(dataset, dtype=float)
        else:
            return source_metrics, "eyeflow_per_segment_metrics"

        computed, _branch, _global, _n_branches, _n_radii, _note = (
            self._compute_block_segment(waveform, beat_periods)
        )
        return computed, "angioeye_recomputed_from_waveforms"

    @staticmethod
    def _pack_region_metrics(
        metrics: dict[str, object],
        vessel: VesselData,
        signal_type: str,
        membership: np.ndarray,
        segment_metrics: dict[str, np.ndarray],
        metric_specs: dict[str, tuple[str, str, str, str]],
    ) -> None:
        n_beats = next(iter(segment_metrics.values())).shape[0]
        for region_index, region_name in enumerate(REGION_NAMES):
            selected = membership[region_index]
            region_prefix = f"{vessel.paths.name}/{region_name}"
            global_prefix = f"{region_prefix}/global"
            branch_indexes = np.flatnonzero(np.any(selected, axis=1))
            for metric_name, values in segment_metrics.items():
                if np.any(selected):
                    region_values = nanmedian(values[:, selected], axis=1)
                else:
                    region_values = np.full(n_beats, np.nan, dtype=np.float32)

                _name, definition, unit, family = metric_specs[metric_name]
                metrics[f"{global_prefix}/{signal_type}/{metric_name}"] = with_attrs(
                    np.asarray(region_values, dtype=np.float32),
                    {
                        "aggregation": (
                            "median over selected branch-radius segment metrics"
                        ),
                        "definition": [definition],
                        "dimDesc": ["beat"],
                        "metric_family": [family],
                        "region": region_name,
                        "signal_type": signal_type,
                        "unit": [unit],
                    },
                )

            for branch_index in branch_indexes:
                branch_id = int(vessel.branch_ids[branch_index])
                branch_selected = selected[branch_index]
                branch_prefix = f"{region_prefix}/by_branch/branch_{branch_id}"

                for metric_name, values in segment_metrics.items():
                    branch_values = nanmedian(
                        values[:, branch_index, branch_selected],
                        axis=1,
                    )
                    _name, definition, unit, family = metric_specs[metric_name]
                    metrics[f"{branch_prefix}/{signal_type}/{metric_name}"] = (
                        with_attrs(
                            np.asarray(branch_values, dtype=np.float32),
                            {
                                "aggregation": (
                                    "median over this branch's selected radius-segment "
                                    "metrics"
                                ),
                                "branch_id": branch_id,
                                "definition": [definition],
                                "dimDesc": ["beat"],
                                "metric_family": [family],
                                "region": region_name,
                                "signal_type": signal_type,
                                "unit": [unit],
                            },
                        )
                    )
