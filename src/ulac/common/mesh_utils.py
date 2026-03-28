import pyvista as pv


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

