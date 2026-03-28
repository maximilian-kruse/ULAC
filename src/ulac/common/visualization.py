import numpy as np
import pyvista as pv

from ulac.construction import constructor

from . import dict_utils


# ==================================================================================================
def plot_2d_mesh(coordinates_2d: np.ndarray, simplices: np.ndarray) -> None:
    z_coordinates = np.zeros(len(coordinates_2d))
    points_3d = np.column_stack((coordinates_2d, z_coordinates))
    mesh = pv.PolyData.from_regular_faces(points_3d, simplices)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, style="wireframe", color="grey")
    plotter.view_xy()
    plotter.show()


def plot_uac_mesh(uac_submesh_data: constructor.UACSubmeshDict) -> None:
    uac_submesh_key_sequences = list(dict_utils.nested_dict_keys(uac_submesh_data))

    plotter = pv.Plotter()
    for key_sequence in uac_submesh_key_sequences:
        uac_submesh = dict_utils.get_dict_entry(key_sequence, uac_submesh_data)
        if uac_submesh is None:
            continue
        coordinates = np.hstack(
            (
                uac_submesh.alpha[:, None],
                uac_submesh.beta[:, None],
                np.zeros((uac_submesh.alpha.shape[0], 1)),
            )
        )
        submesh = pv.PolyData.from_regular_faces(coordinates, uac_submesh.connectivity)
        plotter.add_mesh(submesh, style="wireframe", color="grey")
    plotter.view_xy()
    plotter.show()


def plot_mesh_with_vector_field(
    mesh: pv.PolyData,
    vector_field: np.ndarray,
    scaling_factor: float,
    vector_color: str = "blue",
    mesh_color: str = "lightgray",
) -> None:
    cell_centers = mesh.cell_centers()
    cell_centers.point_data["vector_field_to_plot"] = vector_field
    glyphs = cell_centers.glyph(
        orient="vector_field_to_plot",
        scale=False,
        factor=scaling_factor,
        geom=pv.Arrow(),
    )
    plotter = pv.Plotter()
    plotter.add_mesh(glyphs, color=vector_color)
    plotter.add_mesh(mesh, color=mesh_color)
    plotter.show()


# --------------------------------------------------------------------------------------------------
def plot_mesh_with_segmentation(
    marker_data: constructor.MarkerDict,
    uac_path_data: constructor.UACPathDict,
    mesh: pv.PolyData,
) -> None:
    marker_key_sequences = list(dict_utils.nested_dict_keys(marker_data))
    path_key_sequences = list(dict_utils.nested_dict_keys(uac_path_data))

    plotter = pv.Plotter(window_size=[700, 500])
    plotter.add_mesh(mesh, style="surface", show_edges=True, color="grey")

    for key_sequence in marker_key_sequences:
        marker = dict_utils.get_dict_entry(key_sequence, marker_data)
        if marker is None:
            continue
        marker_mesh = pv.PolyData(mesh.points[marker])
        plotter.add_mesh(marker_mesh, color="red", point_size=15, render_points_as_spheres=True)
    for key_sequence in path_key_sequences:
        path = dict_utils.get_dict_entry(key_sequence, uac_path_data)
        if path is None:
            continue
        path_mesh = pv.PolyData(mesh.points[path.inds])
        path_mesh.point_data["relative_lengths"] = path.relative_lengths
        plotter.add_mesh(
            path_mesh, point_size=10, scalars="relative_lengths", render_points_as_spheres=True
        )

    plotter.show()
