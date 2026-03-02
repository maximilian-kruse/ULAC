from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real


# ==================================================================================================
@dataclass
class MarkerConfig:
    path: Iterable[str]
    position_type: str
    position: int | float


@dataclass
class ParameterizationConfig:
    path: Iterable[str]
    markers: Iterable[Iterable[str]]
    marker_relative_positions: Iterable[Real]


@dataclass
class BoundaryPathConfig:
    feature_tag: str
    coincides_with_mesh_boundary: bool


@dataclass
class ConnectionPathConfig:
    boundary_types: Iterable[str]
    start: Iterable[str]
    end: Iterable[str]
    inadmissible_contact: Iterable[Iterable[str]]
    inadmissible_along: Iterable[Iterable[str]]


@dataclass
class UACConfig:
    path: Iterable[str]
    relative_positions: Iterable[Real]
    uacs: Iterable[tuple[float, float]]


@dataclass
class SubmeshConfig:
    boundary_paths: Iterable[Iterable[str]]
    portions: Iterable[tuple[Real]]
    outside_path: Iterable[str]


@dataclass
class Step:
    id: str
    type: str
    description: str
    apply_to: str | Iterable[str | Iterable[str]]
