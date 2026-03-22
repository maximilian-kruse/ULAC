import numpy as np
import scipy.sparse as sps


# ==================================================================================================
def compute_mean_value_coordinates(
    vertices: np.ndarray,
    simplices: np.ndarray,
    boundary_inds: np.ndarray,
    boundary_coordinates: np.ndarray,
) -> np.ndarray:
    weight_matrix = _compute_weight_matrix(vertices, simplices)
    system_matrix = weight_matrix - sps.eye(weight_matrix.shape[0], format="csr")
    interior_inds = np.setdiff1d(np.unique(simplices), boundary_inds)
    system_matrix_interior = system_matrix[np.ix_(interior_inds, interior_inds)]
    system_matrix_boundary = system_matrix[np.ix_(interior_inds, boundary_inds)]
    rhs_vector_x = system_matrix_boundary @ boundary_coordinates[:, 0]
    rhs_vector_y = system_matrix_boundary @ boundary_coordinates[:, 1]
    uacs_interior_x = sps.linalg.spsolve(system_matrix_interior, -rhs_vector_x)
    uacs_interior_y = sps.linalg.spsolve(system_matrix_interior, -rhs_vector_y)
    uacs_interior = np.hstack([uacs_interior_x[:, None], uacs_interior_y[:, None]])
    uacs = np.zeros((vertices.shape[0], 2))
    uacs[interior_inds] = uacs_interior
    uacs[boundary_inds] = boundary_coordinates

    return uacs


def _compute_weight_matrix(
    vertices: np.ndarray, simplices: np.ndarray
) -> sps.csr_matrix:
    num_vertices_per_simplex = 3
    list_of_weights = []

    for i in range(num_vertices_per_simplex):
        source_vertex = simplices[:, i]
        neighbor_vertex_one = simplices[:, (i + 1) % num_vertices_per_simplex]
        neighbor_vertex_two = simplices[:, (i + 2) % num_vertices_per_simplex]
        source_coordinates = vertices[source_vertex]
        neighbor_one_coordinates = vertices[neighbor_vertex_one]
        neighbor_two_coordinates = vertices[neighbor_vertex_two]
        weight_contrib_one, weight_contrib_two = _compute_weight_contribs(
            source_coordinates, neighbor_one_coordinates, neighbor_two_coordinates
        )
        weights_array_one = np.hstack(
            [weight_contrib_one[:, None], source_vertex[:, None], neighbor_vertex_one[:, None]]
        )
        weights_array_two = np.hstack(
            [weight_contrib_two[:, None], source_vertex[:, None], neighbor_vertex_two[:, None]]
        )
        list_of_weights.append(weights_array_one)
        list_of_weights.append(weights_array_two)

    weight_contrib_array = np.vstack(list_of_weights)
    weight_matrix = _assemble_weight_matrix(weight_contrib_array)
    return weight_matrix


def _compute_weight_contribs(
    source_coordinates: np.ndarray,
    neighbor_one_coordinates: np.ndarray,
    neighbor_two_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edge_one = neighbor_one_coordinates - source_coordinates
    edge_two = neighbor_two_coordinates - source_coordinates
    edge_one_length = np.linalg.norm(edge_one, axis=1)
    edge_two_length = np.linalg.norm(edge_two, axis=1)
    angle_cosine = np.clip(
        np.einsum("ij,ij->i", edge_one, edge_two) / (edge_one_length * edge_two_length),
        -1.0 + 1e-6,
        1.0 - 1e-6,
    )
    half_angle_tangens = np.sqrt((1.0 - angle_cosine) / (1.0 + angle_cosine))
    weight_contrib_one = half_angle_tangens / edge_one_length
    weight_contrib_two = half_angle_tangens / edge_two_length
    return weight_contrib_one, weight_contrib_two


def _assemble_weight_matrix(weight_contribs: np.ndarray) -> sps.coo_matrix:
    weight_matrix = sps.coo_matrix(
        (
            weight_contribs[:, 0],
            (weight_contribs[:, 1].astype(int), weight_contribs[:, 2].astype(int)),
        ),
    )
    weight_matrix.sum_duplicates()
    weight_matrix = weight_matrix.tocsr()
    column_sums = np.array(weight_matrix.sum(axis=1)).flatten()
    column_sum_matrix = sps.diags(1 / column_sums)
    weight_matrix = column_sum_matrix @ weight_matrix
    return weight_matrix
