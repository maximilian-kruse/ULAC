import numpy as np


def compute_cellwise_jacobians(vertices_3d, vertices_2d, simplices):
    edge_3d_one = vertices_3d[simplices[:, 1]] - vertices_3d[simplices[:, 0]]
    edge_3d_two = vertices_3d[simplices[:, 2]] - vertices_3d[simplices[:, 0]]
    edge_2d_one = vertices_2d[simplices[:, 1]] - vertices_2d[simplices[:, 0]]
    edge_2d_two = vertices_2d[simplices[:, 2]] - vertices_2d[simplices[:, 0]]
    jacobians = np.empty((len(simplices), 3, 2))

    for row_ind in range(3):
        jacobians[:, row_ind, 0], jacobians[:, row_ind, 1] = _compute_jacobian_rows(
            edge_3d_one, edge_3d_two, edge_2d_one, edge_2d_two, row_ind
        )
    return jacobians

def _compute_jacobian_rows(edge_3d_one, edge_3d_two, edge_2d_one, edge_2d_two, row_ind):
    j_col_one = (
        edge_3d_one[:, row_ind] * edge_2d_two[:, 0] - edge_3d_two[:, row_ind] * edge_2d_one[:, 0]
    ) / (edge_2d_two[:, 0] * edge_2d_one[:, 1] - edge_2d_two[:, 1] * edge_2d_one[:, 0])
    j_col_two = (edge_3d_one[:, row_ind] - j_col_one * edge_2d_one[:, 1]) / edge_2d_one[:, 0]
    return j_col_one, j_col_two
