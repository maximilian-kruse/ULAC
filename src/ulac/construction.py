import operator
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from functools import reduce

import igl
import numpy as np
import pyvista as pv

from . import configuration as config
from . import construction_base as base
from . import workflow

# ==================================================================================================
type MarkerDict = dict[str, "MarkerDict" | int]
type PathDict = dict[str, "PathDict" | np.ndarray[tuple[int], np.dtype[np.float64]]]
type ParameterizedPathDict = dict[str, "ParameterizedPathDict" | base.ParameterizedPath]
type UACPathDict = dict[str, "UACPathDict" | base.UACPath]
type SubmeshBoundaryDict = dict[
    str, "SubmeshBoundaryDict" | np.ndarray[tuple[int], np.dtype[np.int64]]
]
type SubmeshDict = dict[str, "SubmeshDict" | pv.PolyData]
type UACSubmeshDict = dict[str, "UACSubmeshDict" | pv.PolyData]

type PathConfigDict = dict[
    str, "PathConfigDict" | config.BoundaryPathConfig | config.ConnectionPathConfig
]
type MarkerConfigDict = dict[str, "MarkerConfigDict" | config.MarkerConfig]
type ParameterizationConfigDict = dict[
    str, "ParameterizationConfigDict" | config.ParameterizationConfig
]
type UACConfigDict = dict[str, "UACConfigDict" | config.UACConfig]
type SubmeshConfigDict = dict[str, "SubmeshConfigDict" | config.SubmeshConfig]
type AnyDict = (
    MarkerDict
    | PathDict
    | ParameterizedPathDict
    | UACPathDict
    | MarkerConfigDict
    | PathConfigDict
    | ParameterizationConfigDict
    | UACConfigDict
    | SubmeshConfigDict
)


# --------------------------------------------------------------------------------------------------
def create_empty_dict_from_keys(input_dict: AnyDict) -> dict:
    def _create_empty_dict(d: AnyDict) -> dict:
        if isinstance(d, dict):
            return {key: _create_empty_dict(value) for key, value in d.items()}
        return None

    return _create_empty_dict(input_dict)


# --------------------------------------------------------------------------------------------------
def get_dict_entry(key_sequence: Iterable[str], data_dict: AnyDict) -> object:
    try:
        value = reduce(operator.getitem, key_sequence, data_dict)
    except KeyError as e:
        raise KeyError(f"Key sequence {key_sequence} not found") from e
    return value


# --------------------------------------------------------------------------------------------------
def set_dict_entry(key_sequence: Iterable[str], data_dict: AnyDict, value: object) -> None:
    try:
        reduce(operator.getitem, key_sequence[:-1], data_dict)[key_sequence[-1]] = value
    except KeyError as e:
        raise KeyError(f"Key sequence {key_sequence} not found") from e


# --------------------------------------------------------------------------------------------------
def nested_dict_keys(
    d: dict[str, AnyDict], prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    for key, value in d.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from nested_dict_keys(value, path)
        else:
            yield path


# --------------------------------------------------------------------------------------------------
def print_with_underline(text: str) -> None:
    print(text)
    print("-" * len(text))


# ==================================================================================================
@dataclass
class ULACConstructorSettings:
    mesh: pv.PolyData
    feature_tags: dict[str, int]
    path_config: PathConfigDict
    marker_config: MarkerConfigDict
    parameterization_config: ParameterizationConfigDict
    uac_config: UACConfigDict
    submesh_config: SubmeshConfigDict
    segmentation_workflow: Iterable[workflow.Step]


@dataclass
class ULACData:
    marker_data: MarkerDict | None = None
    raw_path_data: PathDict | None = None
    parameterized_path_data: ParameterizedPathDict | None = None
    uac_path_data: UACPathDict | None = None
    submesh_boundary_data: SubmeshBoundaryDict | None = None
    submesh_data: SubmeshDict | None = None
    uac_submesh_data: UACSubmeshDict | None = None


class ULACConstructor:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: ULACConstructorSettings) -> None:
        self._mesh = settings.mesh
        self._feature_tags = settings.feature_tags
        self._path_config = settings.path_config
        self._marker_config = settings.marker_config
        self._parameterization_config = settings.parameterization_config
        self._uac_config = settings.uac_config
        self._submesh_config = settings.submesh_config
        self._segmentation_workflow = settings.segmentation_workflow

        self._marker_data = create_empty_dict_from_keys(settings.marker_config)
        self._raw_path_data = create_empty_dict_from_keys(settings.path_config)
        self._parameterized_path_data = create_empty_dict_from_keys(
            settings.parameterization_config
        )
        self._uac_path_data = create_empty_dict_from_keys(settings.uac_config)
        self._submesh_boundary_data = create_empty_dict_from_keys(settings.submesh_config)
        self._submesh_data = create_empty_dict_from_keys(settings.submesh_config)
        self._uac_submesh_data = create_empty_dict_from_keys(settings.submesh_config)

    # ----------------------------------------------------------------------------------------------
    def construct_segmentation(self) -> None:
        print("Starting Segmentation")
        print("=====================\n\n")

        for i, step in enumerate(self._segmentation_workflow):
            print_with_underline(f"Step {i + 1}/{len(self._segmentation_workflow)}: {step.id}")
            match step.type:
                case "feature_extraction":
                    self._extract_features(step.apply_to)
                case "marker_extraction":
                    self._extract_markers(step.apply_to)
                case "shortest_path_construction":
                    self._construct_shortest_paths(step.apply_to)
                case "path_parameterization":
                    self._parameterize_paths(step.apply_to)
            print(" ")

    # ----------------------------------------------------------------------------------------------
    def construct_uacs(self) -> None:
        print_with_underline("Construct UAC Paths")
        self._construct_uac_paths()
        print(" ")
        print_with_underline("Extract Submesh Boundaries")
        self._extract_submesh_boundaries()
        print(" ")
        print_with_underline("Extract Submeshes")
        self._extract_submeshes()
        print(" ")
        print_with_underline("Compute UACs on Submeshes")
        self._compute_uacs_on_submeshes()
        print(" ")

    # ----------------------------------------------------------------------------------------------
    @property
    def data(self) -> ULACData:
        ulac_data = ULACData(
            marker_data=self._marker_data,
            raw_path_data=self._raw_path_data,
            parameterized_path_data=self._parameterized_path_data,
            uac_path_data=self._uac_path_data,
            submesh_boundary_data=self._submesh_boundary_data,
            submesh_data=self._submesh_data,
            uac_submesh_data=self._uac_submesh_data,
        )
        return ulac_data

    # ----------------------------------------------------------------------------------------------
    @data.setter
    def data(self, data: ULACData) -> None:
        for input_field, class_attr in zip(
            fields(data),
            (
                "_marker_data",
                "_raw_path_data",
                "_parameterized_path_data",
                "_uac_path_data",
                "_submesh_boundary_data",
                "_submesh_data",
                "_uac_submesh_data",
            ),
            strict=True,
        ):
            input_data = getattr(data, input_field.name)
            if input_data is not None:
                setattr(self, class_attr, input_data)

    # ----------------------------------------------------------------------------------------------
    def _extract_features(self, apply_to: str | Iterable[str]) -> None:
        key_sequences = nested_dict_keys(self._path_config) if apply_to == "all" else apply_to
        for key_sequence in key_sequences:
            try:
                path_config = get_dict_entry(key_sequence, self._path_config)
            except KeyError as e:
                raise KeyError(f"Key sequence {key_sequence} not found in path_config") from e
            if not isinstance(path_config, config.BoundaryPathConfig):
                continue
            print(f"Extracting Feature: {key_sequence}")
            tag_value = self._feature_tags[path_config.feature_tag]
            is_mesh_boundary = path_config.coincides_with_mesh_boundary
            boundary_path = base.get_feature_boundary(self._mesh, tag_value, is_mesh_boundary)
            set_dict_entry(key_sequence, self._raw_path_data, boundary_path)

    # ----------------------------------------------------------------------------------------------
    def _extract_markers(self, apply_to: str | Iterable[str]) -> None:
        key_sequences = nested_dict_keys(self._marker_config) if apply_to == "all" else apply_to
        for key_sequence in key_sequences:
            marker_config = get_dict_entry(key_sequence, self._marker_config)
            print(f"Extracting Marker: {key_sequence}")

            # Get marker from index in raw path
            if marker_config.position_type == "index":
                containing_path = get_dict_entry(marker_config.path, self._raw_path_data)
                if not isinstance(containing_path, np.ndarray):
                    raise ValueError(
                        f"Raw path {marker_config.path} for marker "
                        f"{key_sequence} has not been constructed yet."
                    )
                try:
                    marker_ind = containing_path[marker_config.position]
                except IndexError as e:
                    raise IndexError(
                        f"Index {marker_config.position} out of bounds for path "
                        f"{marker_config.path} with length {len(containing_path)}"
                    ) from e

            # Get marker from relative position in parameterized path
            elif marker_config.position_type == "relative":
                containing_path = get_dict_entry(marker_config.path, self._parameterized_path_data)
                if not isinstance(containing_path, base.ParameterizedPath):
                    raise TypeError(
                        f"Parameterized path {marker_config.path} for marker "
                        f"{key_sequence} has not been constructed yet."
                    )
                try:
                    relative_marker_ind = np.where(
                        containing_path.relative_lengths >= marker_config.position
                    )[0][0]
                    marker_ind = containing_path.inds[relative_marker_ind]
                except IndexError as e:
                    raise ValueError(
                        f"Relative position {marker_config.position} not found on path "
                        f"{marker_config.path}"
                    ) from e
            else:
                raise ValueError(
                    f"Unknown position_type {marker_config.position_type} for marker {key_sequence}"
                )

            # Set marker
            set_dict_entry(key_sequence, self._marker_data, int(marker_ind))

    # ----------------------------------------------------------------------------------------------
    def _construct_shortest_paths(self, apply_to: str | Iterable[str]) -> None:
        key_sequences = nested_dict_keys(self._path_config) if apply_to == "all" else apply_to
        for key_sequence in key_sequences:
            path_config = get_dict_entry(key_sequence, self._path_config)
            if not isinstance(path_config, config.ConnectionPathConfig):
                continue
            print(f"Constructing Shortest Path: {key_sequence}")
            boundaries = []

            # Get boundary sets
            for boundary_type, boundary_id in zip(
                path_config.boundary_types, (path_config.start, path_config.end), strict=True
            ):
                # Option 1: Boundary set is a raw path
                if boundary_type == "path":
                    boundary_path = get_dict_entry(boundary_id, self._raw_path_data)
                    if not isinstance(boundary_path, np.ndarray):
                        raise ValueError(
                            f"Raw path {boundary_id} for path {key_sequence} has not been "
                            "constructed yet."
                        )
                # Option 2: Boundary set is a marker
                elif boundary_type == "marker":
                    boundary_marker = get_dict_entry(boundary_id, self._marker_data)
                    if not isinstance(boundary_marker, int):
                        raise ValueError(
                            f"Marker {boundary_id} for path {key_sequence} has not been "
                            "constructed yet."
                        )
                    boundary_path = np.array((boundary_marker,), dtype=int)
                else:
                    raise ValueError(
                        f"Unknown boundary type {boundary_type} for path {key_sequence}"
                    )
                boundaries.append(boundary_path)

            # Get inadmissible sets
            inadmissible_sets = []
            for inadmissible in (path_config.inadmissible_contact, path_config.inadmissible_along):
                if inadmissible is None:
                    subset = np.array([], dtype=int)
                else:
                    subsets = []
                    for subset_id in inadmissible:
                        subset_path = get_dict_entry(subset_id, self._raw_path_data)
                        if not isinstance(subset_path, np.ndarray):
                            raise TypeError(
                                f"Raw path {subset_id} for path {key_sequence} has not been "
                                "constructed yet."
                            )
                        subsets.append(subset_path)
                    subset = np.unique(np.concatenate(subsets))
                inadmissible_sets.append(subset)

            # Compute shortest Path
            shortest_path = base.construct_shortest_path_between_subsets(
                self._mesh,
                *boundaries,
                *inadmissible_sets,
            )
            set_dict_entry(key_sequence, self._raw_path_data, shortest_path)

    # ----------------------------------------------------------------------------------------------
    def _parameterize_paths(self, apply_to: str | Iterable[str]) -> None:
        key_sequences = (
            nested_dict_keys(self._parameterization_config) if apply_to == "all" else apply_to
        )
        for key_sequence in key_sequences:
            param_config = get_dict_entry(key_sequence, self._parameterization_config)
            print(f"Parameterizing Path: {key_sequence}")

            # Get raw path
            path = get_dict_entry(param_config.path, self._raw_path_data)
            if not isinstance(path, np.ndarray):
                raise TypeError(
                    f"Raw path {key_sequence} for parameterization has not been constructed yet."
                )

            # Get marker data
            marker_inds = []
            for marker_id in param_config.markers:
                marker = get_dict_entry(marker_id, self._marker_data)
                if not isinstance(marker, int):
                    raise TypeError(
                        f"Marker {marker_id} for path {key_sequence} has not been constructed yet."
                    )
                marker_inds.append(marker)

            # Parameterize path
            parameterized_path = base.parameterize_path(
                self._mesh,
                path,
                marker_inds,
                param_config.marker_relative_positions,
            )
            set_dict_entry(key_sequence, self._parameterized_path_data, parameterized_path)

    # ----------------------------------------------------------------------------------------------
    def _construct_uac_paths(self) -> None:
        key_sequences = nested_dict_keys(self._uac_config)
        for key_sequence in key_sequences:
            uac_config = get_dict_entry(key_sequence, self._uac_config)
            print(f"Constructing UACs for Path: {key_sequence}")

            # Get parameterized path
            parameterized_path = get_dict_entry(uac_config.path, self._parameterized_path_data)
            if not isinstance(parameterized_path, base.ParameterizedPath):
                raise TypeError(
                    f"Parameterized path {key_sequence} for UAC construction has not been "
                    "constructed yet."
                )

            # Get uac config data
            relative_positions = uac_config.relative_positions
            uacs = uac_config.uacs

            # Compute UACs
            uac_path = base.compute_uacs_polyline(parameterized_path, relative_positions, uacs)
            set_dict_entry(key_sequence, self._uac_path_data, uac_path)

    # ----------------------------------------------------------------------------------------------
    def _extract_submesh_boundaries(self) -> None:
        key_sequences = nested_dict_keys(self._submesh_config)
        for key_sequence in key_sequences:
            submesh_config = get_dict_entry(key_sequence, self._submesh_config)
            print(f"Extracting boundaries for Submesh: {key_sequence}")

            boundary_inds = []
            boundary_alpha = []
            boundary_beta = []
            for path, portion in zip(
                submesh_config.boundary_paths, submesh_config.portions, strict=True
            ):
                uac_path = get_dict_entry(path, self._uac_path_data)
                if not isinstance(uac_path, base.UACPath):
                    raise TypeError(
                        f"UAC path {uac_path} for submesh {key_sequence} "
                        "has not been constructed yet."
                    )
                relevant_section = np.where(
                    (uac_path.relative_lengths >= portion[0])
                    & (uac_path.relative_lengths <= portion[1])
                )[0]
                boundary_inds.append(uac_path.inds[relevant_section])
                boundary_alpha.append(uac_path.alpha[relevant_section])
                boundary_beta.append(uac_path.beta[relevant_section])

            unique_inds, unique_alpha, unique_beta = self._concatenate_submesh_boundaries(
                boundary_inds, boundary_alpha, boundary_beta
            )
            submesh_boundary = base.UACPath(inds=unique_inds, alpha=unique_alpha, beta=unique_beta)
            set_dict_entry(key_sequence, self._submesh_boundary_data, submesh_boundary)

    # ----------------------------------------------------------------------------------------------
    def _concatenate_submesh_boundaries(
        self,
        boundary_inds: list[np.ndarray],
        boundary_alpha: list[np.ndarray],
        boundary_beta: list[np.ndarray],
    ) -> None:
        unique_inds = boundary_inds.pop(0)
        unique_alpha = boundary_alpha.pop(0)
        unique_beta = boundary_beta.pop(0)

        current_end_point = unique_inds[-1]
        while boundary_inds:
            for i, inds in enumerate(boundary_inds):
                if current_end_point in inds:
                    next_segment_inds = boundary_inds.pop(i)
                    next_segment_alpha = boundary_alpha.pop(i)
                    next_segment_beta = boundary_beta.pop(i)
                    break
            else:
                raise ValueError("Submesh boundary segments do not form a closed loop.")
            if current_end_point == next_segment_inds[-1]:
                next_segment_inds = next_segment_inds[::-1]
                next_segment_alpha = next_segment_alpha[::-1]
                next_segment_beta = next_segment_beta[::-1]
            unique_inds = np.append(unique_inds, next_segment_inds[1:])
            unique_alpha = np.append(unique_alpha, next_segment_alpha[1:])
            unique_beta = np.append(unique_beta, next_segment_beta[1:])
            current_end_point = unique_inds[-1]

        if unique_inds[0] == unique_inds[-1]:
            unique_inds = unique_inds[:-1]
            unique_alpha = unique_alpha[:-1]
            unique_beta = unique_beta[:-1]
        return unique_inds, unique_alpha, unique_beta

    # ----------------------------------------------------------------------------------------------
    def _extract_submeshes(self) -> None:
        key_sequences = nested_dict_keys(self._submesh_config)
        for key_sequence in key_sequences:
            submesh_config = get_dict_entry(key_sequence, self._submesh_config)
            print(f"Extracting Submesh: {key_sequence}")

            # Get boundary path
            submesh_boundary = get_dict_entry(key_sequence, self._submesh_boundary_data)
            if not isinstance(submesh_boundary, base.UACPath):
                raise TypeError(f"Submesh boundary {key_sequence} has not been constructed yet.")

            # Get outside path
            outside_path = get_dict_entry(submesh_config.outside_path, self._raw_path_data)
            if not isinstance(outside_path, np.ndarray):
                raise TypeError(
                    f"Outside path {submesh_config.outside_path} for submesh "
                    f"{key_sequence} has not been constructed yet."
                )

            # Extract submesh
            submesh = base.extract_region_from_boundary(
                self._mesh, submesh_boundary.inds, outside_path
            )
            set_dict_entry(key_sequence, self._submesh_data, submesh)

    # ----------------------------------------------------------------------------------------------
    def _compute_uacs_on_submeshes(self) -> None:
        key_sequences = nested_dict_keys(self._submesh_config)
        for key_sequence in key_sequences:
            print(f"Compute UACs for submesh: {key_sequence}")

            # Get boundary path
            submesh_boundary = get_dict_entry(key_sequence, self._submesh_boundary_data)
            if not isinstance(submesh_boundary, base.UACPath):
                raise TypeError(f"Submesh boundary {key_sequence} has not been constructed yet.")

            # Get submesh
            submesh = get_dict_entry(key_sequence, self._submesh_data)
            if not isinstance(submesh, base.Submesh):
                raise TypeError(f"Submesh {key_sequence} has not been extracted yet.")

            # Compute UACs
            vertices = np.array(self._mesh.points[submesh.inds])
            simplices = submesh.connectivity
            boundary_inds = np.array(
                [np.where(submesh.inds == ind)[0][0] for ind in submesh_boundary.inds]
            )
            uac_coordinates = np.hstack(
                (submesh_boundary.alpha[:, None], submesh_boundary.beta[:, None])
            )
            harmonic_map = igl.harmonic(
                V=vertices,
                F=simplices,
                b=boundary_inds,
                bc=uac_coordinates,
                k=1,
            )
            uac_submesh = base.UACSubmesh(
                inds=submesh.inds,
                connectivity=simplices,
                cell_inds=submesh.cell_inds,
                alpha=harmonic_map[:, 0],
                beta=harmonic_map[:, 1],
            )
            set_dict_entry(key_sequence, self._uac_submesh_data, uac_submesh)
