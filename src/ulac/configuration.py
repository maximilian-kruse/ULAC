from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np


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


# ==================================================================================================
path_configs = {
    # ----------------------------------------------------------------------------------------------
    "LIPV": {
        "inner": BoundaryPathConfig(
            feature_tag="LIPV",
            coincides_with_mesh_boundary=False,
        ),
        "outer": BoundaryPathConfig(
            feature_tag="LIPV",
            coincides_with_mesh_boundary=True,
        ),
        "anterior_posterior": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LIPV", "inner", "anterior_posterior"],
            end=["LIPV", "outer"],
            inadmissible_contact=None,
            inadmissible_along=[["LIPV", "inner"], ["LIPV", "outer"]],
        ),
        "septal_lateral": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LIPV", "inner", "septal_lateral"],
            end=["LIPV", "outer"],
            inadmissible_contact=[["LIPV", "anterior_posterior"]],
            inadmissible_along=[["LIPV", "inner"], ["LIPV", "outer"]],
        ),
        "anchor": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LIPV", "inner", "anchor"],
            end=["LIPV", "outer"],
            inadmissible_contact=[["LIPV", "anterior_posterior"], ["LIPV", "septal_lateral"]],
            inadmissible_along=[["LIPV", "inner"], ["LIPV", "outer"]],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "LSPV": {
        "inner": BoundaryPathConfig(
            feature_tag="LSPV",
            coincides_with_mesh_boundary=False,
        ),
        "outer": BoundaryPathConfig(
            feature_tag="LSPV",
            coincides_with_mesh_boundary=True,
        ),
        "anterior_posterior": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LSPV", "inner", "anterior_posterior"],
            end=["LSPV", "outer"],
            inadmissible_contact=None,
            inadmissible_along=[["LSPV", "inner"], ["LSPV", "outer"]],
        ),
        "septal_lateral": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LSPV", "inner", "septal_lateral"],
            end=["LSPV", "outer"],
            inadmissible_contact=[["LSPV", "anterior_posterior"]],
            inadmissible_along=[["LSPV", "inner"], ["LSPV", "outer"]],
        ),
        "anchor": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["LSPV", "inner", "anchor"],
            end=["LSPV", "outer"],
            inadmissible_contact=[["LSPV", "anterior_posterior"], ["LSPV", "septal_lateral"]],
            inadmissible_along=[["LSPV", "inner"], ["LSPV", "outer"]],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "RSPV": {
        "inner": BoundaryPathConfig(
            feature_tag="RSPV",
            coincides_with_mesh_boundary=False,
        ),
        "outer": BoundaryPathConfig(
            feature_tag="RSPV",
            coincides_with_mesh_boundary=True,
        ),
        "anterior_posterior": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RSPV", "inner", "anterior_posterior"],
            end=["RSPV", "outer"],
            inadmissible_contact=None,
            inadmissible_along=[["RSPV", "inner"], ["RSPV", "outer"]],
        ),
        "septal_lateral": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RSPV", "inner", "septal_lateral"],
            end=["RSPV", "outer"],
            inadmissible_contact=[["RSPV", "anterior_posterior"]],
            inadmissible_along=[["RSPV", "inner"], ["RSPV", "outer"]],
        ),
        "anchor": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RSPV", "inner", "anchor"],
            end=["RSPV", "outer"],
            inadmissible_contact=[["RSPV", "anterior_posterior"], ["RSPV", "septal_lateral"]],
            inadmissible_along=[["RSPV", "inner"], ["RSPV", "outer"]],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "RIPV": {
        "inner": BoundaryPathConfig(
            feature_tag="RIPV",
            coincides_with_mesh_boundary=False,
        ),
        "outer": BoundaryPathConfig(
            feature_tag="RIPV",
            coincides_with_mesh_boundary=True,
        ),
        "anterior_posterior": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RIPV", "inner", "anterior_posterior"],
            end=["RIPV", "outer"],
            inadmissible_contact=None,
            inadmissible_along=[["RIPV", "inner"], ["RIPV", "outer"]],
        ),
        "septal_lateral": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RIPV", "inner", "septal_lateral"],
            end=["RIPV", "outer"],
            inadmissible_contact=[["RIPV", "anterior_posterior"]],
            inadmissible_along=[["RIPV", "inner"], ["RIPV", "outer"]],
        ),
        "anchor": ConnectionPathConfig(
            boundary_types=["marker", "path"],
            start=["RIPV", "inner", "anchor"],
            end=["RIPV", "outer"],
            inadmissible_contact=[["RIPV", "anterior_posterior"], ["RIPV", "septal_lateral"]],
            inadmissible_along=[["RIPV", "inner"], ["RIPV", "outer"]],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "LAA": BoundaryPathConfig(
        feature_tag="LAA",
        coincides_with_mesh_boundary=False,
    ),
    # ----------------------------------------------------------------------------------------------
    "MV": BoundaryPathConfig(
        feature_tag="MV",
        coincides_with_mesh_boundary=True,
    ),
    # ----------------------------------------------------------------------------------------------
    "roof": {
        "LIPV_LSPV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LIPV", "inner"],
            end=["LSPV", "inner"],
            inadmissible_contact=None,
            inadmissible_along=[["LIPV", "inner"], ["LSPV", "inner"]],
        ),
        "LSPV_RSPV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LSPV", "inner"],
            end=["RSPV", "inner"],
            inadmissible_contact=None,
            inadmissible_along=[["LSPV", "inner"], ["RSPV", "inner"]],
        ),
        "RSPV_RIPV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["RSPV", "inner"],
            end=["RIPV", "inner"],
            inadmissible_contact=None,
            inadmissible_along=[["RSPV", "inner"], ["RIPV", "inner"]],
        ),
        "RIPV_LIPV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["RIPV", "inner"],
            end=["LIPV", "inner"],
            inadmissible_contact=None,
            inadmissible_along=[["RIPV", "inner"], ["LIPV", "inner"]],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "anchor": {
        "LIPV_LAA": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LIPV", "inner"],
            end=["LAA"],
            inadmissible_contact=None,
            inadmissible_along=[["LIPV", "inner"], ["LAA"]],
        ),
        "LAA_MV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LAA"],
            end=["MV"],
            inadmissible_contact=None,
            inadmissible_along=[["LAA"], ["MV"]],
        ),
        "LSPV_MV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["LSPV", "inner"],
            end=["MV"],
            inadmissible_contact=(["LAA"], ["anchor", "LIPV_LAA"], ["anchor", "LAA_MV"]),
            inadmissible_along=[["LSPV", "inner"], ["MV"]],
        ),
        "RSPV_MV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["RSPV", "inner"],
            end=["MV"],
            inadmissible_contact=None,
            inadmissible_along=[["RSPV", "inner"], ["MV"]],
        ),
        "RIPV_MV": ConnectionPathConfig(
            boundary_types=["path", "path"],
            start=["RIPV", "inner"],
            end=["MV"],
            inadmissible_contact=(["anchor", "RSPV_MV"],),
            inadmissible_along=[["RIPV", "inner"], ["MV"]],
        ),
        "LAA_lateral": ConnectionPathConfig(
            boundary_types=["path", "marker"],
            start=["LAA"],
            end=["anchor", "LSPV_MV"],
            inadmissible_contact=(["MV"],),
            inadmissible_along=[["LAA"], ["anchor", "LSPV_MV"]],
        ),
        "LAA_posterior": ConnectionPathConfig(
            boundary_types=["path", "marker"],
            start=["LAA"],
            end=["anchor", "RIPV_MV"],
            inadmissible_contact=(["MV"],),
            inadmissible_along=[["LAA"], ["anchor", "RIPV_MV"]],
        ),
    },
}


# ==================================================================================================
PV_SEPTAL_LATERAL = 1 / 4
PV_ANCHOR = 5 / 8


parameterization_configs = {
    # ----------------------------------------------------------------------------------------------
    "LIPV": {
        "inner": ParameterizationConfig(
            path=["LIPV", "inner"],
            markers=[
                ["LIPV", "inner", "anterior_posterior"],
                ["LIPV", "inner", "septal_lateral"],
                ["LIPV", "inner", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "outer": ParameterizationConfig(
            path=["LIPV", "outer"],
            markers=[
                ["LIPV", "outer", "anterior_posterior"],
                ["LIPV", "outer", "septal_lateral"],
                ["LIPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "anterior_posterior": ParameterizationConfig(
            path=["LIPV", "anterior_posterior"],
            markers=[
                ["LIPV", "inner", "anterior_posterior"],
                ["LIPV", "outer", "anterior_posterior"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "septal_lateral": ParameterizationConfig(
            path=["LIPV", "septal_lateral"],
            markers=[
                ["LIPV", "inner", "septal_lateral"],
                ["LIPV", "outer", "septal_lateral"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "anchor": ParameterizationConfig(
            path=["LIPV", "anchor"],
            markers=[
                ["LIPV", "inner", "anchor"],
                ["LIPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "LSPV": {
        "inner": ParameterizationConfig(
            path=["LSPV", "inner"],
            markers=[
                ["LSPV", "inner", "anterior_posterior"],
                ["LSPV", "inner", "septal_lateral"],
                ["LSPV", "inner", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "outer": ParameterizationConfig(
            path=["LSPV", "outer"],
            markers=[
                ["LSPV", "outer", "anterior_posterior"],
                ["LSPV", "outer", "septal_lateral"],
                ["LSPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "anterior_posterior": ParameterizationConfig(
            path=["LSPV", "anterior_posterior"],
            markers=[
                ["LSPV", "inner", "anterior_posterior"],
                ["LSPV", "outer", "anterior_posterior"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "septal_lateral": ParameterizationConfig(
            path=["LSPV", "septal_lateral"],
            markers=[
                ["LSPV", "inner", "septal_lateral"],
                ["LSPV", "outer", "septal_lateral"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "anchor": ParameterizationConfig(
            path=["LSPV", "anchor"],
            markers=[
                ["LSPV", "inner", "anchor"],
                ["LSPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "RSPV": {
        "inner": ParameterizationConfig(
            path=["RSPV", "inner"],
            markers=[
                ["RSPV", "inner", "anterior_posterior"],
                ["RSPV", "inner", "septal_lateral"],
                ["RSPV", "inner", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "outer": ParameterizationConfig(
            path=["RSPV", "outer"],
            markers=[
                ["RSPV", "outer", "anterior_posterior"],
                ["RSPV", "outer", "septal_lateral"],
                ["RSPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "anterior_posterior": ParameterizationConfig(
            path=["RSPV", "anterior_posterior"],
            markers=[
                ["RSPV", "inner", "anterior_posterior"],
                ["RSPV", "outer", "anterior_posterior"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "septal_lateral": ParameterizationConfig(
            path=["RSPV", "septal_lateral"],
            markers=[
                ["RSPV", "inner", "septal_lateral"],
                ["RSPV", "outer", "septal_lateral"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "anchor": ParameterizationConfig(
            path=["RSPV", "anchor"],
            markers=[
                ["RSPV", "inner", "anchor"],
                ["RSPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "RIPV": {
        "inner": ParameterizationConfig(
            path=["RIPV", "inner"],
            markers=[
                ["RIPV", "inner", "anterior_posterior"],
                ["RIPV", "inner", "septal_lateral"],
                ["RIPV", "inner", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "outer": ParameterizationConfig(
            path=["RIPV", "outer"],
            markers=[
                ["RIPV", "outer", "anterior_posterior"],
                ["RIPV", "outer", "septal_lateral"],
                ["RIPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR],
        ),
        "anterior_posterior": ParameterizationConfig(
            path=["RIPV", "anterior_posterior"],
            markers=[
                ["RIPV", "inner", "anterior_posterior"],
                ["RIPV", "outer", "anterior_posterior"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "septal_lateral": ParameterizationConfig(
            path=["RIPV", "septal_lateral"],
            markers=[
                ["RIPV", "inner", "septal_lateral"],
                ["RIPV", "outer", "septal_lateral"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "anchor": ParameterizationConfig(
            path=["RIPV", "anchor"],
            markers=[
                ["RIPV", "inner", "anchor"],
                ["RIPV", "outer", "anchor"],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "LAA": ParameterizationConfig(
        path=["LAA"],
        markers=[
            ["LAA", "LIPV"],
            ["LAA", "lateral"],
            ["LAA", "MV"],
            ["LAA", "posterior"],
        ],
        marker_relative_positions=[0, 1 / 4, 1 / 2, 3 / 4],
    ),
    # ----------------------------------------------------------------------------------------------
    "MV": ParameterizationConfig(
        path=["MV"],
        markers=[
            ["MV", "RSPV"],
            ["MV", "LSPV"],
            ["MV", "LAA"],
            ["MV", "RIPV"],
        ],
        marker_relative_positions=[0, 1 / 4, 1 / 2, 3 / 4],
    ),
    # ----------------------------------------------------------------------------------------------
    "roof": {
        "LIPV_LSPV": ParameterizationConfig(
            path=["roof", "LIPV_LSPV"],
            markers=[
                ["LIPV", "inner", "anterior_posterior"],
                ["LSPV", "inner", "anterior_posterior"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "LSPV_RSPV": ParameterizationConfig(
            path=["roof", "LSPV_RSPV"],
            markers=[
                ["LSPV", "inner", "septal_lateral"],
                ["RSPV", "inner", "septal_lateral"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "RSPV_RIPV": ParameterizationConfig(
            path=["roof", "RSPV_RIPV"],
            markers=[
                ["RSPV", "inner", "anterior_posterior"],
                ["RIPV", "inner", "anterior_posterior"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "RIPV_LIPV": ParameterizationConfig(
            path=["roof", "RIPV_LIPV"],
            markers=[
                ["RIPV", "inner", "septal_lateral"],
                ["LIPV", "inner", "septal_lateral"],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
    # ----------------------------------------------------------------------------------------------
    "anchor": {
        "LIPV_LAA": ParameterizationConfig(
            path=["anchor", "LIPV_LAA"],
            markers=[
                ["LIPV", "inner", "anchor"],
                ["LAA", "LIPV"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "LAA_MV": ParameterizationConfig(
            path=["anchor", "LAA_MV"],
            markers=[
                ["LAA", "MV"],
                ["MV", "LAA"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "LSPV_MV_anterior": ParameterizationConfig(
            path=["anchor", "LSPV_MV"],
            markers=[
                ["LSPV", "inner", "anchor"],
                ["MV", "LSPV"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "LSPV_MV_lateral": ParameterizationConfig(
            path=["anchor", "LSPV_MV"],
            markers=[
                ["LSPV", "inner", "anchor"],
                ["anchor", "LSPV_MV"],
                ["MV", "LSPV"],
            ],
            marker_relative_positions=[0, 1 / 2, 1],
        ),
        "RSPV_MV": ParameterizationConfig(
            path=["anchor", "RSPV_MV"],
            markers=[
                ["RSPV", "inner", "anchor"],
                ["MV", "RSPV"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "RIPV_MV_septal": ParameterizationConfig(
            path=["anchor", "RIPV_MV"],
            markers=[
                ["RIPV", "inner", "anchor"],
                ["MV", "RIPV"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "RIPV_MV_posterior": ParameterizationConfig(
            path=["anchor", "RIPV_MV"],
            markers=[
                ["RIPV", "inner", "anchor"],
                ["anchor", "RIPV_MV"],
                ["MV", "RIPV"],
            ],
            marker_relative_positions=[0, 1 / 2, 1],
        ),
        "LAA_lateral": ParameterizationConfig(
            path=["anchor", "LAA_lateral"],
            markers=[
                ["LAA", "lateral"],
                ["anchor", "LSPV_MV"],
            ],
            marker_relative_positions=[0, 1],
        ),
        "LAA_posterior": ParameterizationConfig(
            path=["anchor", "LAA_posterior"],
            markers=[
                ["LAA", "posterior"],
                ["anchor", "RIPV_MV"],
            ],
            marker_relative_positions=[0, 1],
        ),
    },
}


# ==================================================================================================
marker_configs = {
    # ----------------------------------------------------------------------------------------------
    "LIPV": {
        "inner": {
            "anterior_posterior": MarkerConfig(
                path=["roof", "LIPV_LSPV"],
                position_type="index",
                position=0,
            ),
            "septal_lateral": MarkerConfig(
                path=["roof", "RIPV_LIPV"],
                position_type="index",
                position=-1,
            ),
            "anchor": MarkerConfig(
                path=["anchor", "LIPV_LAA"],
                position_type="index",
                position=0,
            ),
        },
        "outer": {
            "anterior_posterior": MarkerConfig(
                path=["LIPV", "anterior_posterior"],
                position_type="index",
                position=-1,
            ),
            "septal_lateral": MarkerConfig(
                path=["LIPV", "septal_lateral"],
                position_type="index",
                position=-1,
            ),
            "anchor": MarkerConfig(
                path=["LIPV", "anchor"],
                position_type="index",
                position=-1,
            ),
        },
    },
    "LSPV": {
        "inner": {
            "anterior_posterior": MarkerConfig(
                path=["roof", "LIPV_LSPV"],
                position_type="index",
                position=-1,
            ),
            "septal_lateral": MarkerConfig(
                path=["roof", "LSPV_RSPV"],
                position_type="index",
                position=0,
            ),
            "anchor": MarkerConfig(
                path=["anchor", "LSPV_MV"],
                position_type="index",
                position=0,
            ),
        },
        "outer": {
            "anterior_posterior": MarkerConfig(
                path=["LSPV", "anterior_posterior"],
                position_type="index",
                position=-1,
            ),
            "septal_lateral": MarkerConfig(
                path=["LSPV", "septal_lateral"],
                position_type="index",
                position=-1,
            ),
            "anchor": MarkerConfig(
                path=["LSPV", "anchor"],
                position_type="index",
                position=-1,
            ),
        },
    },
    "RSPV": {
        "inner": {
            "anterior_posterior": MarkerConfig(
                path=["roof", "RSPV_RIPV"],
                position_type="index",
                position=0,
            ),
            "septal_lateral": MarkerConfig(
                path=["roof", "LSPV_RSPV"],
                position_type="index",
                position=-1,
            ),
            "anchor": MarkerConfig(
                path=["anchor", "RSPV_MV"],
                position_type="index",
                position=0,
            ),
        },
        "outer": {
            "anterior_posterior": MarkerConfig(
                path=["RSPV", "anterior_posterior"],
                position_type="index",
                position=-1,
            ),
            "septal_lateral": MarkerConfig(
                path=["RSPV", "septal_lateral"],
                position_type="index",
                position=-1,
            ),
            "anchor": MarkerConfig(
                path=["RSPV", "anchor"],
                position_type="index",
                position=-1,
            ),
        },
    },
    "RIPV": {
        "inner": {
            "anterior_posterior": MarkerConfig(
                path=["roof", "RSPV_RIPV"],
                position_type="index",
                position=-1,
            ),
            "septal_lateral": MarkerConfig(
                path=["roof", "RIPV_LIPV"],
                position_type="index",
                position=0,
            ),
            "anchor": MarkerConfig(
                path=["anchor", "RIPV_MV"],
                position_type="index",
                position=0,
            ),
        },
        "outer": {
            "anterior_posterior": MarkerConfig(
                path=["RIPV", "anterior_posterior"],
                position_type="index",
                position=-1,
            ),
            "septal_lateral": MarkerConfig(
                path=["RIPV", "septal_lateral"],
                position_type="index",
                position=-1,
            ),
            "anchor": MarkerConfig(
                path=["RIPV", "anchor"],
                position_type="index",
                position=-1,
            ),
        },
    },
    # ----------------------------------------------------------------------------------------------
    "LAA": {
        "LIPV": MarkerConfig(
            path=["anchor", "LIPV_LAA"],
            position_type="index",
            position=-1,
        ),
        "MV": MarkerConfig(
            path=["anchor", "LAA_MV"],
            position_type="index",
            position=0,
        ),
        "lateral": MarkerConfig(
            path=["anchor", "LAA_lateral"],
            position_type="index",
            position=0,
        ),
        "posterior": MarkerConfig(
            path=["anchor", "LAA_posterior"],
            position_type="index",
            position=0,
        ),
    },
    "MV": {
        "RSPV": MarkerConfig(
            path=["anchor", "RSPV_MV"],
            position_type="index",
            position=-1,
        ),
        "LSPV": MarkerConfig(
            path=["anchor", "LSPV_MV"],
            position_type="index",
            position=-1,
        ),
        "LAA": MarkerConfig(
            path=["anchor", "LAA_MV"],
            position_type="index",
            position=-1,
        ),
        "RIPV": MarkerConfig(
            path=["anchor", "RIPV_MV"],
            position_type="index",
            position=-1,
        ),
    },
    "anchor": {
        "LSPV_MV": MarkerConfig(
            path=["anchor", "LSPV_MV_anterior"],
            position_type="relative",
            position=0.5,
        ),
        "RIPV_MV": MarkerConfig(
            path=["anchor", "RIPV_MV_septal"],
            position_type="relative",
            position=0.5,
        ),
    },
}


# ==================================================================================================
LIPV_CENTER = (2, 3 / 2)
LSPV_CENTER = (2, 1)
RIPV_CENTER = (1, 3 / 2)
RSPV_CENTER = (1, 1)
LAA_CENTER = (13 / 5, 21 / 10)
PV_INNER_RADIUS = 1 / 5
PV_OUTER_RADIUS = 1 / 10
LAA_LENGTH = 4 / 5
LAA_DISTANCE = 1 / 4
LAA_DEPTH = 1 / 4
ANCHOR_LENGTH = 1


# ==================================================================================================
uac_configs = {
    "LIPV": {
        "inner": UACConfig(
            path=["LIPV", "inner"],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR, 1],
            uacs=[
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
                (LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
            ],
        ),
        "outer": UACConfig(
            path=["LIPV", "outer"],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR, 1],
            uacs=[
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS),
                (LIPV_CENTER[0] - PV_OUTER_RADIUS, LIPV_CENTER[1]),
                (
                    LIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS),
            ],
        ),
        "anterior_posterior": UACConfig(
            path=["LIPV", "anterior_posterior"],
            relative_positions=[0, 1],
            uacs=[
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_OUTER_RADIUS),
            ],
        ),
        "septal_lateral": UACConfig(
            path=["LIPV", "septal_lateral"],
            relative_positions=[0, 1],
            uacs=[
                (LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
                (LIPV_CENTER[0] - PV_OUTER_RADIUS, LIPV_CENTER[1]),
            ],
        ),
        "anchor": UACConfig(
            path=["LIPV", "anchor"],
            relative_positions=[0, 1],
            uacs=[
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    LIPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
    },
    "LSPV": {
        "inner": UACConfig(
            path=["LSPV", "inner"],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR, 1],
            uacs=[
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
                (LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
                (
                    LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
            ],
        ),
        "outer": UACConfig(
            path=["LSPV", "outer"],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR, 1],
            uacs=[
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS),
                (LSPV_CENTER[0] - PV_OUTER_RADIUS, LSPV_CENTER[1]),
                (
                    LSPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS),
            ],
        ),
        "anterior_posterior": UACConfig(
            path=["LSPV", "anterior_posterior"],
            relative_positions=[0, 1],
            uacs=[
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_OUTER_RADIUS),
            ],
        ),
        "septal_lateral": UACConfig(
            path=["LSPV", "septal_lateral"],
            relative_positions=[0, 1],
            uacs=[
                (LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
                (LSPV_CENTER[0] - PV_OUTER_RADIUS, LSPV_CENTER[1]),
            ],
        ),
        "anchor": UACConfig(
            path=["LSPV", "anchor"],
            relative_positions=[0, 1],
            uacs=[
                (
                    LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    LSPV_CENTER[0] + PV_OUTER_RADIUS / np.sqrt(2),
                    LSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
    },
    "RSPV": {
        "inner": UACConfig(
            path=["RSPV", "inner"],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR, 1],
            uacs=[
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
                (RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1]),
                (
                    RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
            ],
        ),
        "outer": UACConfig(
            path=["RSPV", "outer"],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR, 1],
            uacs=[
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS),
                (RSPV_CENTER[0] + PV_OUTER_RADIUS, RSPV_CENTER[1]),
                (
                    RSPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS),
            ],
        ),
        "anterior_posterior": UACConfig(
            path=["RSPV", "anterior_posterior"],
            relative_positions=[0, 1],
            uacs=[
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_OUTER_RADIUS),
            ],
        ),
        "septal_lateral": UACConfig(
            path=["RSPV", "septal_lateral"],
            relative_positions=[0, 1],
            uacs=[
                (RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1]),
                (RSPV_CENTER[0] + PV_OUTER_RADIUS, RSPV_CENTER[1]),
            ],
        ),
        "anchor": UACConfig(
            path=["RSPV", "anchor"],
            relative_positions=[0, 1],
            uacs=[
                (
                    RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    RSPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RSPV_CENTER[1] - PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
    },
    "RIPV": {
        "inner": UACConfig(
            path=["RIPV", "inner"],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR, 1],
            uacs=[
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
                (RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
                (
                    RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
            ],
        ),
        "outer": UACConfig(
            path=["RIPV", "outer"],
            relative_positions=[0, PV_SEPTAL_LATERAL, PV_ANCHOR, 1],
            uacs=[
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS),
                (RIPV_CENTER[0] + PV_OUTER_RADIUS, RIPV_CENTER[1]),
                (
                    RIPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS),
            ],
        ),
        "anterior_posterior": UACConfig(
            path=["RIPV", "anterior_posterior"],
            relative_positions=[0, 1],
            uacs=[
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_OUTER_RADIUS),
            ],
        ),
        "septal_lateral": UACConfig(
            path=["RIPV", "septal_lateral"],
            relative_positions=[0, 1],
            uacs=[
                (RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
                (RIPV_CENTER[0] + PV_OUTER_RADIUS, RIPV_CENTER[1]),
            ],
        ),
        "anchor": UACConfig(
            path=["RIPV", "anchor"],
            relative_positions=[0, 1],
            uacs=[
                (
                    RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    RIPV_CENTER[0] - PV_OUTER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_OUTER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
    },
    "LAA": {
        "LAA": UACConfig(
            path=["LAA"],
            relative_positions=[0, 1 / 4, 1 / 2, 3 / 4, 1],
            uacs=[
                (LAA_CENTER[0] - LAA_LENGTH / 2, LAA_CENTER[1] - LAA_LENGTH / 2),
                (LAA_CENTER[0] + LAA_LENGTH / 2, LAA_CENTER[1] - LAA_LENGTH / 2),
                (LAA_CENTER[0] + LAA_LENGTH / 2, LAA_CENTER[1] + LAA_LENGTH / 2),
                (LAA_CENTER[0] - LAA_LENGTH / 2, LAA_CENTER[1] + LAA_LENGTH / 2),
                (LAA_CENTER[0] - LAA_LENGTH / 2, LAA_CENTER[1] - LAA_LENGTH / 2),
            ],
        ),
        "lateral": UACConfig(
            path=["LAA"],
            relative_positions=[0, 1 / 4, 1 / 2],
            uacs=[
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + LAA_DISTANCE,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + 2 * LAA_DISTANCE,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) - LAA_DEPTH,
                ),
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + 3 * LAA_DISTANCE,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
        "posterior": UACConfig(
            path=["LAA"],
            relative_positions=[1 / 2, 3 / 4, 1],
            uacs=[
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + 3 * LAA_DISTANCE,
                ),
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) - LAA_DEPTH,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + 2 * LAA_DISTANCE,
                ),
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + LAA_DISTANCE,
                ),
            ],
        ),
    },
    "MV": {
        "anterior": UACConfig(
            path=["MV"],
            relative_positions=[0, 1 / 4],
            uacs=[
                (RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), 0),
                (LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), 0),
            ],
        ),
        "lateral": UACConfig(
            path=["MV"],
            relative_positions=[1 / 4, 1 / 2],
            uacs=[
                (LSPV_CENTER[0] + ANCHOR_LENGTH, LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2)),
                (LIPV_CENTER[0] + ANCHOR_LENGTH, LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2)),
            ],
        ),
        "posterior": UACConfig(
            path=["MV"],
            relative_positions=[1 / 2, 3 / 4],
            uacs=[
                (LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + ANCHOR_LENGTH),
                (RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), RIPV_CENTER[1] + ANCHOR_LENGTH),
            ],
        ),
        "septal": UACConfig(
            path=["MV"],
            relative_positions=[3 / 4, 1],
            uacs=[
                (0, RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2)),
                (0, RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2)),
            ],
        ),
    },
    "roof": {
        "LIPV_LSPV": UACConfig(
            path=["roof", "LIPV_LSPV"],
            relative_positions=[0, 1],
            uacs=[
                (LIPV_CENTER[0], LIPV_CENTER[1] - PV_INNER_RADIUS),
                (LSPV_CENTER[0], LSPV_CENTER[1] + PV_INNER_RADIUS),
            ],
        ),
        "LSPV_RSPV": UACConfig(
            path=["roof", "LSPV_RSPV"],
            relative_positions=[0, 1],
            uacs=[
                (LSPV_CENTER[0] - PV_INNER_RADIUS, LSPV_CENTER[1]),
                (RSPV_CENTER[0] + PV_INNER_RADIUS, RSPV_CENTER[1]),
            ],
        ),
        "RSPV_RIPV": UACConfig(
            path=["roof", "RSPV_RIPV"],
            relative_positions=[0, 1],
            uacs=[
                (RSPV_CENTER[0], RSPV_CENTER[1] + PV_INNER_RADIUS),
                (RIPV_CENTER[0], RIPV_CENTER[1] - PV_INNER_RADIUS),
            ],
        ),
        "RIPV_LIPV": UACConfig(
            path=["roof", "RIPV_LIPV"],
            relative_positions=[0, 1],
            uacs=[
                (RIPV_CENTER[0] + PV_INNER_RADIUS, RIPV_CENTER[1]),
                (LIPV_CENTER[0] - PV_INNER_RADIUS, LIPV_CENTER[1]),
            ],
        ),
    },
    "anchor": {
        "LIPV_LAA": {
            "lateral": UACConfig(
                path=["anchor", "LIPV_LAA"],
                relative_positions=[0, 1],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + LAA_DISTANCE,
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            "posterior": UACConfig(
                path=["anchor", "LIPV_LAA"],
                relative_positions=[0, 1],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + LAA_DISTANCE,
                    ),
                ],
            ),
        },
        "LAA_MV": {
            "lateral": UACConfig(
                path=["anchor", "LAA_MV"],
                relative_positions=[0, 1],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + 3 * LAA_DISTANCE,
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        LIPV_CENTER[0] + ANCHOR_LENGTH,
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
            "posterior": UACConfig(
                path=["anchor", "LAA_MV"],
                relative_positions=[0, 1],
                uacs=[
                    (
                        LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + 3 * LAA_DISTANCE,
                    ),
                    (LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), LIPV_CENTER[1] + ANCHOR_LENGTH),
                ],
            ),
        },
        "LSPV_MV": {
            "anterior": UACConfig(
                path=["anchor", "LSPV_MV_anterior"],
                relative_positions=[0, 1],
                uacs=[
                    (
                        LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2), 0),
                ],
            ),
            "lateral": UACConfig(
                path=["anchor", "LSPV_MV_lateral"],
                relative_positions=[0, 1 / 2, 1],
                uacs=[
                    (
                        LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2),
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + ANCHOR_LENGTH / 2,
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        LSPV_CENTER[0] + ANCHOR_LENGTH,
                        LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                ],
            ),
        },
        "RSPV_MV": {
            "anterior": UACConfig(
                path=["anchor", "RSPV_MV"],
                relative_positions=[0, 1],
                uacs=[
                    (
                        RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2), 0),
                ],
            ),
            "septal": UACConfig(
                path=["anchor", "RSPV_MV"],
                relative_positions=[0, 1],
                uacs=[
                    (
                        RSPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (0, RSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2)),
                ],
            ),
        },
        "RIPV_MV": {
            "septal": UACConfig(
                path=["anchor", "RIPV_MV_septal"],
                relative_positions=[0, 1],
                uacs=[
                    (
                        RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (0, RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2)),
                ],
            ),
            "posterior": UACConfig(
                path=["anchor", "RIPV_MV_posterior"],
                relative_positions=[0, 1 / 2, 1],
                uacs=[
                    (
                        RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2),
                    ),
                    (
                        RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + ANCHOR_LENGTH / 2,
                    ),
                    (
                        RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                        RIPV_CENTER[1] + ANCHOR_LENGTH,
                    ),
                ],
            ),
        },
        "LAA_lateral": UACConfig(
            path=["anchor", "LAA_lateral"],
            relative_positions=[0, 1],
            uacs=[
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + 2 * LAA_DISTANCE,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) - LAA_DEPTH,
                ),
                (
                    LSPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) + ANCHOR_LENGTH / 2,
                    LSPV_CENTER[1] - PV_INNER_RADIUS / np.sqrt(2),
                ),
            ],
        ),
        "LAA_posterior": UACConfig(
            path=["anchor", "LAA_posterior"],
            relative_positions=[0, 1],
            uacs=[
                (
                    LIPV_CENTER[0] + PV_INNER_RADIUS / np.sqrt(2) - LAA_DEPTH,
                    LIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + 2 * LAA_DISTANCE,
                ),
                (
                    RIPV_CENTER[0] - PV_INNER_RADIUS / np.sqrt(2),
                    RIPV_CENTER[1] + PV_INNER_RADIUS / np.sqrt(2) + ANCHOR_LENGTH / 2,
                ),
            ],
        ),
    },
}


# ==================================================================================================
submesh_configs = {
    "roof": SubmeshConfig(
        boundary_paths=[
            ["LIPV", "inner"],
            ["LSPV", "inner"],
            ["RSPV", "inner"],
            ["RIPV", "inner"],
            ["roof", "LIPV_LSPV"],
            ["roof", "LSPV_RSPV"],
            ["roof", "RSPV_RIPV"],
            ["roof", "RIPV_LIPV"],
        ],
        portions=[
            (0, PV_SEPTAL_LATERAL),
            (0, PV_SEPTAL_LATERAL),
            (0, PV_SEPTAL_LATERAL),
            (0, PV_SEPTAL_LATERAL),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
        ],
        outside_path=["MV"],
    ),
    "anterior": SubmeshConfig(
        boundary_paths=[
            ["LSPV", "inner"],
            ["RSPV", "inner"],
            ["roof", "LSPV_RSPV"],
            ["anchor", "LSPV_MV", "anterior"],
            ["anchor", "RSPV_MV", "anterior"],
            ["MV", "anterior"],
        ],
        portions=[
            (PV_SEPTAL_LATERAL, PV_ANCHOR),
            (PV_SEPTAL_LATERAL, PV_ANCHOR),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
        ],
        outside_path=["LIPV", "inner"],
    ),
    "septal": SubmeshConfig(
        boundary_paths=[
            ["RIPV", "inner"],
            ["RSPV", "inner"],
            ["roof", "RSPV_RIPV"],
            ["anchor", "RIPV_MV", "septal"],
            ["anchor", "RSPV_MV", "septal"],
            ["MV", "septal"],
        ],
        portions=[(PV_ANCHOR, 1), (PV_ANCHOR, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
        outside_path=["LIPV", "inner"],
    ),
    "posterior_roof": SubmeshConfig(
        boundary_paths=[
            ["RIPV", "inner"],
            ["LIPV", "inner"],
            ["roof", "RIPV_LIPV"],
            ["anchor", "RIPV_MV", "posterior"],
            ["anchor", "LIPV_LAA", "posterior"],
            ["anchor", "LAA_posterior"],
            ["LAA", "posterior"],
        ],
        portions=[
            (PV_SEPTAL_LATERAL, PV_ANCHOR),
            (PV_SEPTAL_LATERAL, PV_ANCHOR),
            (0, 1),
            (0, 1 / 2),
            (0, 1),
            (0, 1),
            (3 / 4, 1),
        ],
        outside_path=["LSPV", "inner"],
    ),
    "posterior_mv": SubmeshConfig(
        boundary_paths=[
            ["anchor", "RIPV_MV", "posterior"],
            ["anchor", "LAA_MV", "posterior"],
            ["anchor", "LAA_posterior"],
            ["LAA", "posterior"],
            ["MV", "posterior"],
        ],
        portions=[(1 / 2, 1), (0, 1), (0, 1), (1 / 2, 3 / 4), (0, 1)],
        outside_path=["LSPV", "inner"],
    ),
    "lateral_roof": SubmeshConfig(
        boundary_paths=[
            ["LIPV", "inner"],
            ["LSPV", "inner"],
            ["roof", "LIPV_LSPV"],
            ["anchor", "LIPV_LAA", "lateral"],
            ["anchor", "LSPV_MV", "lateral"],
            ["anchor", "LAA_lateral"],
            ["LAA", "lateral"],
        ],
        portions=[(PV_ANCHOR, 1), (PV_ANCHOR, 1), (0, 1), (0, 1), (0, 1 / 2), (0, 1), (0, 1 / 4)],
        outside_path=["RIPV", "inner"],
    ),
    "lateral_mv": SubmeshConfig(
        boundary_paths=[
            ["anchor", "LAA_MV", "lateral"],
            ["anchor", "LSPV_MV", "lateral"],
            ["anchor", "LAA_lateral"],
            ["LAA", "lateral"],
            ["MV", "lateral"],
        ],
        portions=[(0, 1), (1 / 2, 1), (0, 1), (1 / 4, 1 / 2), (0, 1)],
        outside_path=["RIPV", "inner"],
    ),
    "laa": SubmeshConfig(
        boundary_paths=[["LAA", "LAA"]],
        portions=[(0, 1)],
        outside_path=["RIPV", "inner"],
    ),
    "LIPV": {
        "segment_1": SubmeshConfig(
            boundary_paths=[
                ["LIPV", "inner"],
                ["LIPV", "outer"],
                ["LIPV", "anterior_posterior"],
                ["LIPV", "septal_lateral"],
            ],
            portions=[
                (0, PV_SEPTAL_LATERAL),
                (0, PV_SEPTAL_LATERAL),
                (0, 1),
                (0, 1),
            ],
            outside_path=["MV"],
        ),
        "segment_2": SubmeshConfig(
            boundary_paths=[
                ["LIPV", "inner"],
                ["LIPV", "outer"],
                ["LIPV", "septal_lateral"],
                ["LIPV", "anchor"],
            ],
            portions=[
                (PV_SEPTAL_LATERAL, PV_ANCHOR),
                (PV_SEPTAL_LATERAL, PV_ANCHOR),
                (0, 1),
                (0, 1),
            ],
            outside_path=["MV"],
        ),
        "segment_3": SubmeshConfig(
            boundary_paths=[
                ["LIPV", "inner"],
                ["LIPV", "outer"],
                ["LIPV", "anchor"],
                ["LIPV", "anterior_posterior"],
            ],
            portions=[
                (PV_ANCHOR, 1),
                (PV_ANCHOR, 1),
                (0, 1),
                (0, 1),
            ],
            outside_path=["MV"],
        ),
    },
    "LSPV": {
        "segment_1": SubmeshConfig(
            boundary_paths=[
                ["LSPV", "inner"],
                ["LSPV", "outer"],
                ["LSPV", "anterior_posterior"],
                ["LSPV", "septal_lateral"],
            ],
            portions=[
                (0, PV_SEPTAL_LATERAL),
                (0, PV_SEPTAL_LATERAL),
                (0, 1),
                (0, 1),
            ],
            outside_path=["MV"],
        ),
        "segment_2": SubmeshConfig(
            boundary_paths=[
                ["LSPV", "inner"],
                ["LSPV", "outer"],
                ["LSPV", "septal_lateral"],
                ["LSPV", "anchor"],
            ],
            portions=[
                (PV_SEPTAL_LATERAL, PV_ANCHOR),
                (PV_SEPTAL_LATERAL, PV_ANCHOR),
                (0, 1),
                (0, 1),
            ],
            outside_path=["MV"],
        ),
        "segment_3": SubmeshConfig(
            boundary_paths=[
                ["LSPV", "inner"],
                ["LSPV", "outer"],
                ["LSPV", "anchor"],
                ["LSPV", "anterior_posterior"],
            ],
            portions=[
                (PV_ANCHOR, 1),
                (PV_ANCHOR, 1),
                (0, 1),
                (0, 1),
            ],
            outside_path=["MV"],
        ),
    },
    "RSPV": {
        "segment_1": SubmeshConfig(
            boundary_paths=[
                ["RSPV", "inner"],
                ["RSPV", "outer"],
                ["RSPV", "anterior_posterior"],
                ["RSPV", "septal_lateral"],
            ],
            portions=[
                (0, PV_SEPTAL_LATERAL),
                (0, PV_SEPTAL_LATERAL),
                (0, 1),
                (0, 1),
            ],
            outside_path=["MV"],
        ),
        "segment_2": SubmeshConfig(
            boundary_paths=[
                ["RSPV", "inner"],
                ["RSPV", "outer"],
                ["RSPV", "septal_lateral"],
                ["RSPV", "anchor"],
            ],
            portions=[
                (PV_SEPTAL_LATERAL, PV_ANCHOR),
                (PV_SEPTAL_LATERAL, PV_ANCHOR),
                (0, 1),
                (0, 1),
            ],
            outside_path=["MV"],
        ),
        "segment_3": SubmeshConfig(
            boundary_paths=[
                ["RSPV", "inner"],
                ["RSPV", "outer"],
                ["RSPV", "anchor"],
                ["RSPV", "anterior_posterior"],
            ],
            portions=[
                (PV_ANCHOR, 1),
                (PV_ANCHOR, 1),
                (0, 1),
                (0, 1),
            ],
            outside_path=["MV"],
        ),
    },
    "RIPV": {
        "segment_1": SubmeshConfig(
            boundary_paths=[
                ["RIPV", "inner"],
                ["RIPV", "outer"],
                ["RIPV", "anterior_posterior"],
                ["RIPV", "septal_lateral"],
            ],
            portions=[
                (0, PV_SEPTAL_LATERAL),
                (0, PV_SEPTAL_LATERAL),
                (0, 1),
                (0, 1),
            ],
            outside_path=["MV"],
        ),
        "segment_2": SubmeshConfig(
            boundary_paths=[
                ["RIPV", "inner"],
                ["RIPV", "outer"],
                ["RIPV", "septal_lateral"],
                ["RIPV", "anchor"],
            ],
            portions=[
                (PV_SEPTAL_LATERAL, PV_ANCHOR),
                (PV_SEPTAL_LATERAL, PV_ANCHOR),
                (0, 1),
                (0, 1),
            ],
            outside_path=["MV"],
        ),
        "segment_3": SubmeshConfig(
            boundary_paths=[
                ["RIPV", "inner"],
                ["RIPV", "outer"],
                ["RIPV", "anchor"],
                ["RIPV", "anterior_posterior"],
            ],
            portions=[
                (PV_ANCHOR, 1),
                (PV_ANCHOR, 1),
                (0, 1),
                (0, 1),
            ],
            outside_path=["MV"],
        ),
    },
}
