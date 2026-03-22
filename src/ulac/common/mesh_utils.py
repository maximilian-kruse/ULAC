import numpy as np
import pyvista as pv

from ulac.construction import constructor

from . import dict_utils


# ==================================================================================================
def convert_unstructured_to_polydata_mesh(mesh: pv.UnstructuredGrid) -> pv.PolyData:
    point_data = mesh.point_data
    cell_data = mesh.cell_data
    polydata_mesh = pv.PolyData(mesh.points, mesh.cells)
    for key in point_data:
        polydata_mesh.point_data[key] = point_data[key]
    for key in cell_data:
        polydata_mesh.cell_data[key] = cell_data[key]

    return polydata_mesh


# --------------------------------------------------------------------------------------------------
def visualize_segmentation(
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


# --------------------------------------------------------------------------------------------------
def visualize_uac_mesh(uac_submesh_data: constructor.UACSubmeshDict) -> None:
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
