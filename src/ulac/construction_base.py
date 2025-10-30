from dataclasses import dataclass

import igraph as ig
import numpy as np
import pyvista as pv
import trimesh as tm
from numba import njit
from numba.typed import List


# ==================================================================================================
@dataclass
class ParameterizedPath:
    inds: np.ndarray = None
    relative_lengths: np.ndarray = None


@dataclass
class UACPath(ParameterizedPath):
    alpha: np.ndarray = None
    beta: np.ndarray = None


@dataclass
class Submesh:
    inds: np.ndarray = None
    connectivity: np.ndarray = None
    cell_inds: np.ndarray = None


@dataclass
class UACSubmesh(Submesh):
    alpha: np.ndarray = None
    beta: np.ndarray = None


# ==================================================================================================
def get_feature_boundary(
    mesh: pv.PolyData, feature_tag: int, coincides_with_geometry_boundary: bool
) -> tuple[np.ndarray, np.ndarray]:
    feature_mesh = mesh.extract_values(feature_tag, scalars="anatomical_tags")
    feature_boundaries = feature_mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    geometry_boundaries = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    feature_boundary_inds = feature_mesh.point_data["vtkOriginalPointIds"][
        feature_boundaries.point_data["vtkOriginalPointIds"]
    ]
    geometry_boundary_inds = geometry_boundaries.point_data["vtkOriginalPointIds"]

    if coincides_with_geometry_boundary:
        feature_boundary_inds = np.intersect1d(feature_boundary_inds, geometry_boundary_inds)
    else:
        feature_boundary_inds = np.setdiff1d(feature_boundary_inds, geometry_boundary_inds)

    ordered_boundary_inds = _construct_ordered_path_from_indices(mesh, feature_boundary_inds)
    return ordered_boundary_inds


# --------------------------------------------------------------------------------------------------
def _construct_ordered_path_from_indices(mesh: pv.PolyData, path_indices: np.ndarray) -> np.ndarray:
    tm_mesh = tm.Trimesh(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    path_edges = tm_mesh.edges[np.isin(tm_mesh.edges, path_indices).all(axis=1)].flatten()
    local_edges = np.array([np.where(path_indices == ind)[0][0] for ind in path_edges])
    local_edges = local_edges.reshape(-1, 2)
    graph = ig.Graph(local_edges, directed=False)
    ordered_path, _ = graph.dfs(0)
    return path_indices[ordered_path]


# --------------------------------------------------------------------------------------------------
def construct_shortest_path_between_subsets(
    mesh: pv.PolyData,
    subset_one: np.ndarray,
    subset_two: np.ndarray,
    inadmissible_contact_set: np.ndarray,
    inadmissible_along_set: np.ndarray,
):
    tm_mesh = tm.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:])
    edges = tm_mesh.edges_unique
    edge_lengths = tm_mesh.edges_unique_length
    inadmissible_contact_edges = np.where(np.isin(edges, inadmissible_contact_set).any(axis=1))[0]
    inadmissible_along_edges = np.where(np.isin(edges, inadmissible_along_set).all(axis=1))[0]
    inadmissible_edges = np.unique(
        np.concatenate([inadmissible_contact_edges, inadmissible_along_edges])
    )
    admissible_edges = np.delete(edges, inadmissible_edges, axis=0)
    admissible_edges_lengths = np.delete(edge_lengths, inadmissible_edges, axis=0)

    graph = ig.Graph(admissible_edges, directed=False)
    distances = np.array(graph.distances(subset_one, subset_two, weights=admissible_edges_lengths))
    rel_start_point, rel_end_point = np.unravel_index(np.argmin(distances), distances.shape)
    graph = ig.Graph(admissible_edges, directed=False)
    shortest_path = np.array(
        graph.get_shortest_paths(
            subset_one[rel_start_point],
            to=subset_two[rel_end_point],
            weights=admissible_edges_lengths,
        )[0]
    )
    return np.array(shortest_path, dtype=int)


# --------------------------------------------------------------------------------------------------
def extract_region_from_boundary(
    mesh: pv.PolyData, boundary_inds: np.ndarray, outside_inds: np.ndarray
) -> np.ndarray:
    mesh_points_without_boundary = np.setdiff1d(np.arange(mesh.number_of_points), boundary_inds)
    mesh_without_boundary = mesh.extract_points(mesh_points_without_boundary, adjacent_cells=False)
    submeshes = mesh_without_boundary.split_bodies()
    coinciding_outside_inds = np.where(
        submeshes[0].point_data["vtkOriginalPointIds"] == outside_inds[0]
    )[0]
    inside_mesh = submeshes[0] if coinciding_outside_inds.size == 0 else submeshes[1]
    seed_point = inside_mesh.point_data["vtkOriginalPointIds"][0]
    seed_ind = np.where(np.isin(mesh.faces.reshape(-1, 4)[:, 1:4], seed_point).any(axis=1))[0][0]

    tm_mesh = tm.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:4])
    face_adjacency = tm_mesh.face_adjacency
    face_adjacency_flipped = np.flip(face_adjacency, axis=1)
    face_adjacency_complete = np.vstack([face_adjacency, face_adjacency_flipped])
    sorting_mask = np.argsort(face_adjacency_complete[:, 0])
    face_adjacency_sorted = face_adjacency_complete[sorting_mask]
    _, new_face_start_inds = np.unique(face_adjacency_sorted[:, 0], return_index=True)
    neighbor_faces = face_adjacency_sorted[:, 1]
    boundary_edges = np.hstack([boundary_inds[:-1, None], boundary_inds[1:, None]])
    boundary_edges = np.append(boundary_edges, [[boundary_inds[-1], boundary_inds[0]]], axis=0)

    submesh_cell_inds = _extract_region_from_boundary(
        tm_mesh.faces,
        seed_ind,
        boundary_edges,
        new_face_start_inds,
        neighbor_faces,
    )
    pv_submesh = mesh.extract_cells(submesh_cell_inds)
    vertex_inds = pv_submesh.point_data["vtkOriginalPointIds"]
    connectivity = np.array(pv_submesh.cells.reshape(-1, 4)[:, 1:])
    submesh = Submesh(inds=vertex_inds, connectivity=connectivity, cell_inds=submesh_cell_inds)

    return submesh


# --------------------------------------------------------------------------------------------------
@njit
def _extract_region_from_boundary(
    faces: np.ndarray,
    seed_ind: int,
    boundary_edges: np.ndarray,
    new_face_start_inds: np.ndarray,
    neighbor_faces: np.ndarray,
) -> np.ndarray:
    visited_cells = np.zeros(new_face_start_inds.shape[0], dtype=np.bool_)
    active_list = List([seed_ind])

    while active_list:
        current_cell_ind = active_list.pop()
        current_cell = faces[current_cell_ind]
        visited_cells[current_cell_ind] = True
        start_ind = new_face_start_inds[current_cell_ind]
        end_ind = (
            new_face_start_inds[current_cell_ind + 1]
            if current_cell_ind + 1 < new_face_start_inds.size
            else neighbor_faces.size
        )
        for i in range(start_ind, end_ind):
            neighbor_cell_ind = neighbor_faces[i]
            neighbor_cell = faces[neighbor_cell_ind]
            shared_edge = np.intersect1d(current_cell, neighbor_cell)
            for boundary_edge in boundary_edges:
                if  shared_edge[0] in boundary_edge and shared_edge[1] in boundary_edge:
                    is_boundary = True
                    break
            else:
                is_boundary = False
            if not is_boundary and not visited_cells[neighbor_cell_ind]:
                active_list.append(neighbor_cell_ind)

    return np.where(visited_cells)[0]


# ==================================================================================================
def parameterize_path(
    mesh: pv.PolyData, path: np.ndarray, marker_inds: list[int], marker_values: list[float]
) -> ParameterizedPath:
    reordered_path, relative_marker_inds = _reorder_path_by_markers(
        path, marker_inds, marker_values
    )
    parameterized_path = _parameterize_by_relative_length(
        mesh, reordered_path, relative_marker_inds, marker_values
    )
    return parameterized_path


# --------------------------------------------------------------------------------------------------
def _reorder_path_by_markers(
    path: np.ndarray,
    marker_inds: list[int],
    marker_values: list[float],
) -> tuple[np.ndarray, list[int]]:
    relative_start_ind_location = np.where(path == marker_inds[0])[0]
    ordered_path = np.roll(path, -relative_start_ind_location)

    relative_marker_inds = [np.where(ordered_path == ind)[0][0] for ind in marker_inds]
    marker_ind_order = np.argsort(relative_marker_inds)
    marker_values_order = np.argsort(marker_values)
    if not np.array_equal(marker_ind_order, marker_values_order):
        ordered_path = np.roll(np.flip(ordered_path), 1)
        relative_marker_inds = [np.where(ordered_path == ind)[0][0] for ind in marker_inds]
    return ordered_path, relative_marker_inds


# --------------------------------------------------------------------------------------------------
def _parameterize_by_relative_length(
    mesh: pv.PolyData, path: np.ndarray, relative_marker_inds: list[int], marker_values: list[float]
) -> ParameterizedPath:
    coordinates = np.array(mesh.points[path])
    if 1.0 not in marker_values:
        path = np.append(path, path[0])
        relative_marker_inds = [*relative_marker_inds, path.size - 1]
        marker_values = [*marker_values, 1.0]
        coordinates = np.append(coordinates, [coordinates[0]], axis=0)
    edge_lengths = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
    cumulative_lengths = np.cumsum(edge_lengths)
    cumulative_lengths = np.insert(cumulative_lengths, 0, 0.0)
    segmented_points = np.zeros(path.size)

    num_segments = len(relative_marker_inds) - 1
    for i in range(num_segments):
        start_ind = relative_marker_inds[i]
        end_ind = relative_marker_inds[i + 1]
        start_value = marker_values[i]
        end_value = marker_values[i + 1]
        shifted_cumulative_lengths = (
            cumulative_lengths[start_ind : end_ind + 1] - cumulative_lengths[start_ind]
        )
        relative_lengths = (
            start_value
            + (end_value - start_value)
            * shifted_cumulative_lengths
            / shifted_cumulative_lengths[-1]
        )
        segmented_points[start_ind : end_ind + 1] = relative_lengths

    parameterized_path = ParameterizedPath(path, segmented_points)
    return parameterized_path


# ==================================================================================================
def compute_uacs_polyline(
    path: ParameterizedPath,
    segment_points: list[float],
    segment_uacs: list[tuple[float, float]],
) -> UACPath:
    alpha_values, beta_values, ind_values, relative_lengths = [], [], [], []
    num_segments = len(segment_points) - 1

    for i in range(num_segments):
        start_ind = np.where(np.isclose(path.relative_lengths, segment_points[i]))[0][0]
        end_ind = np.where(np.isclose(path.relative_lengths, segment_points[i + 1]))[0][0]
        segment_inds = path.inds[start_ind : end_ind + 1]
        start_uacs = segment_uacs[i]
        end_uacs = segment_uacs[i + 1]
        path_segment = path.relative_lengths[start_ind : end_ind + 1]
        length_range = np.max(path_segment) - np.min(path_segment)
        scaled_length = (path_segment - np.min(path_segment)) / length_range
        alpha = start_uacs[0] + (end_uacs[0] - start_uacs[0]) * scaled_length
        beta = start_uacs[1] + (end_uacs[1] - start_uacs[1]) * scaled_length

        if i < num_segments - 1:
            alpha = alpha[:-1]
            beta = beta[:-1]
            segment_inds = segment_inds[:-1]
            path_segment = path_segment[:-1]
        alpha_values.append(alpha)
        beta_values.append(beta)
        ind_values.append(segment_inds)
        relative_lengths.append(path_segment)

    ind_values = np.concatenate(ind_values)
    relative_lengths = np.concatenate(relative_lengths)
    alpha_values = np.concatenate(alpha_values)
    beta_values = np.concatenate(beta_values)
    uac_path = UACPath(
        inds=ind_values, relative_lengths=relative_lengths, alpha=alpha_values, beta=beta_values
    )
    return uac_path
