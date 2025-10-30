from collections.abc import Iterable
from dataclasses import dataclass


# ==================================================================================================
@dataclass
class Step:
    id: str
    type: str
    description: str
    apply_to: str | Iterable[str | Iterable[str]]


# ==================================================================================================

workflow = [
    # ----------------------------------------------------------------------------------------------
    # Initial step: extract boundaries from tags and mesh boundaries
    Step(
        id="boundary_extraction",
        type="feature_extraction",
        description="""Extract feature an MV boundaries from anatomical tags and mesh boundaries""",
        apply_to="all",
    ),
    # ----------------------------------------------------------------------------------------------
    # Construct roof and anchor paths
    Step(
        id="path_construction_roof_anchors",
        type="shortest_path_construction",
        description="""Construct shortest PV connection paths, LIPV->LSPV->RSPV->RIPV->LIPV,
                     anchor paths LIPV->LAA, LAA->MV, LSPV->MV, RSPV->MV, RIPV->MV""",
        apply_to=[
            ("roof", "LIPV_LSPV"),
            ("roof", "LSPV_RSPV"),
            ("roof", "RSPV_RIPV"),
            ("roof", "RIPV_LIPV"),
            ("anchor", "LIPV_LAA"),
            ("anchor", "LAA_MV"),
            ("anchor", "LSPV_MV"),
            ("anchor", "RSPV_MV"),
            ("anchor", "RIPV_MV"),
        ],
    ),
    # ----------------------------------------------------------------------------------------------
    # Extract markers from roof and anchor paths
    Step(
        id="markers_roof_anchors",
        type="marker_extraction",
        description="""Get markers from roof and anchor points""",
        apply_to=[
            ("LIPV", "inner", "anterior_posterior"),
            ("LIPV", "inner", "septal_lateral"),
            ("LIPV", "inner", "anchor"),
            ("LSPV", "inner", "anterior_posterior"),
            ("LSPV", "inner", "septal_lateral"),
            ("LSPV", "inner", "anchor"),
            ("RSPV", "inner", "anterior_posterior"),
            ("RSPV", "inner", "septal_lateral"),
            ("RSPV", "inner", "anchor"),
            ("RIPV", "inner", "anterior_posterior"),
            ("RIPV", "inner", "septal_lateral"),
            ("RIPV", "inner", "anchor"),
            ("LAA", "LIPV"),
            ("LAA", "MV"),
            ("MV", "LAA"),
            ("MV", "LSPV"),
            ("MV", "RSPV"),
            ("MV", "RIPV"),
        ],
    ),
    # ----------------------------------------------------------------------------------------------
    # Construct pv segment paths from new markers
    Step(
        id="path_construction_pv_segments",
        type="shortest_path_construction",
        description="""Construct pv segment paths from new markers""",
        apply_to=[
            ("LIPV", "anterior_posterior"),
            ("LIPV", "septal_lateral"),
            ("LIPV", "anchor"),
            ("LSPV", "anterior_posterior"),
            ("LSPV", "septal_lateral"),
            ("LSPV", "anchor"),
            ("RSPV", "anterior_posterior"),
            ("RSPV", "septal_lateral"),
            ("RSPV", "anchor"),
            ("RIPV", "anterior_posterior"),
            ("RIPV", "septal_lateral"),
            ("RIPV", "anchor"),
        ]
    ),
    # ----------------------------------------------------------------------------------------------
    # Extract markers from pv segment paths
    Step(
        id="markers_pv_segments",
        type="marker_extraction",
        description="""Get PV outer markers from PV segments""",
        apply_to=[
            ("LIPV", "outer", "anterior_posterior"),
            ("LIPV", "outer", "septal_lateral"),
            ("LIPV", "outer", "anchor"),
            ("LSPV", "outer", "anterior_posterior"),
            ("LSPV", "outer", "septal_lateral"),
            ("LSPV", "outer", "anchor"),
            ("RSPV", "outer", "anterior_posterior"),
            ("RSPV", "outer", "septal_lateral"),
            ("RSPV", "outer", "anchor"),
            ("RIPV", "outer", "anterior_posterior"),
            ("RIPV", "outer", "septal_lateral"),
            ("RIPV", "outer", "anchor"),
        ],
    ),
    # ----------------------------------------------------------------------------------------------
    # Parameterize all available paths
    Step(
        id="path_parameterization_existing",
        type="path_parameterization",
        description="""Parameterize available paths""",
        apply_to=[
            ("LIPV", "inner"),
            ("LIPV", "outer"),
            ("LIPV", "anterior_posterior"),
            ("LIPV", "septal_lateral"),
            ("LIPV", "anchor"),
            ("LSPV", "inner"),
            ("LSPV", "outer"),
            ("LSPV", "anterior_posterior"),
            ("LSPV", "septal_lateral"),
            ("LSPV", "anchor"),
            ("RSPV", "inner"),
            ("RSPV", "outer"),
            ("RSPV", "anterior_posterior"),
            ("RSPV", "septal_lateral"),
            ("RSPV", "anchor"),
            ("RIPV", "inner"),
            ("RIPV", "outer"),
            ("RIPV", "anterior_posterior"),
            ("RIPV", "septal_lateral"),
            ("RIPV", "anchor"),
            ("MV",),
            ("roof", "LIPV_LSPV"),
            ("roof", "LSPV_RSPV"),
            ("roof", "RSPV_RIPV"),
            ("roof", "RIPV_LIPV"),
            ("anchor", "LIPV_LAA"),
            ("anchor", "LAA_MV"),
            ("anchor", "RSPV_MV"),
            ("anchor", "LSPV_MV_anterior"),
            ("anchor", "RIPV_MV_septal"),
        ],
    ),
    # ----------------------------------------------------------------------------------------------
    # Extract LAA lateral and posterior markers on LSPV and RIPV
    Step(
        id="markers_laa_anchors",
        type="marker_extraction",
        description="""Get markers for LAA anchors on LSPV and RIPV""",
        apply_to=[
            ("anchor", "LSPV_MV"),
            ("anchor", "RIPV_MV"),
        ],
    ),
    # ----------------------------------------------------------------------------------------------
    # Construct LAA lateral and posterior anchor paths
    Step(
        id="path_construction_laa_anchors",
        description="""Construct LAA lateral and posterior anchors""",
        type="shortest_path_construction",
        apply_to=[
            ("anchor", "LAA_lateral"),
            ("anchor", "LAA_posterior"),
        ]
    ),
    # ----------------------------------------------------------------------------------------------
    # Extract LAA lateral and posterior markers on LAA
    Step(
        id="markers_laa_anchors",
        description="""Get markers for LAA lateral and posterior anchors""",
        type="marker_extraction",
        apply_to=[
            ("LAA", "lateral"),
            ("LAA", "posterior"),
        ]
    ),
    # ----------------------------------------------------------------------------------------------
    # Parameterize remaining paths
    Step(
        id="path_parameterization_remaining",
        description="""Parameterize LAA and its lateral/posterior anchors""",
        type="path_parameterization",
        apply_to=[
            ("LAA",),
            ("anchor", "LAA_lateral"),
            ("anchor", "LAA_posterior"),
            ("anchor", "LSPV_MV_lateral"),
            ("anchor", "RIPV_MV_posterior"),
        ]
    ),
]
