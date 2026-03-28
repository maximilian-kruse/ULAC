from pathlib import Path

import igl
import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmeshpy
import numpy as np
import pyvista as pv

from .common import jacobian, mvc


def construct_cellwise_basis(
    mesh: pv.PolyData, exterior_boundary_tag: int, exterior_boundary_radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    simplices = np.array(mesh.faces.reshape(-1, 4)[:, 1:4])
    mvc_coordinates = _construct_smooth_parameterization(
        mesh, exterior_boundary_tag, exterior_boundary_radius
    )
    cellwise_jacobians = jacobian.compute_cellwise_jacobians(
        vertices_3d=mesh.points, vertices_2d=mvc_coordinates, simplices=simplices
    )
    basis_vectors = _orthonormalize_jacobian_columns(cellwise_jacobians)
    return basis_vectors, mvc_coordinates, cellwise_jacobians


def _construct_smooth_parameterization(
    mesh: pv.PolyData, exterior_boundary_tag: int, exterior_boundary_radius: float
) -> np.ndarray:
    vertices = np.array(mesh.points)
    simplices = np.array(mesh.faces.reshape(-1, 4)[:, 1:4])
    exterior_loop = _find_exterior_boundary(mesh, exterior_boundary_tag)
    extended_vertices, extended_simplices = _create_mesh_with_holes_filled(
        simplices, vertices, exterior_loop
    )
    boundary_x_coords, boundary_y_coords = _map_path_to_circle(
        radius=exterior_boundary_radius, path=exterior_loop
    )
    exterior_boundary_coords = np.stack((boundary_x_coords, boundary_y_coords), axis=1)
    mvc_coordinates = mvc.compute_mean_value_coordinates(
        extended_vertices, extended_simplices, exterior_loop, exterior_boundary_coords
    )
    mvc_coordinates = mvc_coordinates[: len(vertices)]
    return mvc_coordinates


def _find_exterior_boundary(mesh: pv.PolyData, exterior_boundary_tag: int) -> np.ndarray:
    simplices = np.array(mesh.faces.reshape(-1, 4)[:, 1:4])
    exterior_boundary_tags = mesh.extract_values(exterior_boundary_tag, scalars="anatomical_tags")
    exterior_boundary_inds = mesh.point_data["vtkOriginalPointIds"][
        exterior_boundary_tags.point_data["vtkOriginalPointIds"]
    ]
    boundary_loops = igl.boundary_loop_all(simplices)

    interior_loops = []
    exterior_loop = None
    for boundary_loop in boundary_loops:
        if set(boundary_loop).intersection(set(exterior_boundary_inds)):
            exterior_loop = boundary_loop
        else:
            interior_loops.append(boundary_loop)
    assert exterior_loop is not None, "No exterior loop found in the mesh."
    assert len(interior_loops) + 1 == len(boundary_loops), (
        "More than one exterior loop found in the mesh."
    )
    return exterior_loop


def _create_mesh_with_holes_filled(
    simplices: np.ndarray, vertices: np.ndarray, exterior_boundary_inds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    meshlib_mesh = mrmeshnumpy.meshFromFacesVerts(simplices, vertices)
    hole_edges = meshlib_mesh.topology.findHoleRepresentiveEdges()
    exterior_boundary_ind = _find_exterior_edge_id(meshlib_mesh, hole_edges, exterior_boundary_inds)
    del hole_edges[exterior_boundary_ind]
    params = mrmeshpy.FillHoleParams()
    new_faces_bitset = mrmeshpy.FaceBitSet()
    params.outNewFaces = new_faces_bitset
    for edge_id in hole_edges:
        mrmeshpy.fillHole(meshlib_mesh, edge_id, params)
    extended_vertices = mrmeshnumpy.getNumpyVerts(meshlib_mesh)
    extended_simplices = mrmeshnumpy.getNumpyFaces(meshlib_mesh.topology)
    return extended_vertices, extended_simplices


def _find_exterior_edge_id(
    meshlib_mesh: mrmeshpy.Mesh, hole_edges: list, exterior_boundary_inds: np.ndarray
) -> int:
    for i, edge_id in enumerate(hole_edges):
        if int(meshlib_mesh.topology.org(edge_id)) in exterior_boundary_inds:
            return i
    raise ValueError("No edge found with origin in the exterior boundary.")


def _map_path_to_circle(radius: float, path: Path) -> tuple[np.ndarray, np.ndarray]:
    x_coordinates = np.zeros(len(path))
    y_coordinates = np.zeros(len(path))

    for i, _ in enumerate(path):
        angle = i / len(path) * 2 * np.pi
        x_coordinates[i] = radius * np.cos(angle)
        y_coordinates[i] = radius * np.sin(angle)

    return x_coordinates, y_coordinates


def _orthonormalize_jacobian_columns(cellwise_jacobians: np.ndarray) -> np.ndarray:
    first_basis_vector = cellwise_jacobians[..., 0]
    second_basis_vector = cellwise_jacobians[..., 1]

    first_basis_vector = first_basis_vector / np.linalg.norm(
        first_basis_vector, axis=1, keepdims=True
    )
    second_basis_vector = (
        second_basis_vector
        - np.einsum("ni,ni->n", first_basis_vector, second_basis_vector)[:, None]
        * first_basis_vector
    )
    second_basis_vector = second_basis_vector / np.linalg.norm(
        second_basis_vector, axis=1, keepdims=True
    )
    return np.stack([first_basis_vector, second_basis_vector], axis=-1)

