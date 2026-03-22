import numpy as np
import scipy.spatial as sps

from ulac.common import dict_utils
from ulac.construction import constructor

from . import jacobian

# ==================================================================================================
type UACScalarDict = dict[str, "UACScalarDict" | np.ndarray]
type UACVectorDict = dict[str, "UACVectorDict" | np.ndarray]
type UACJacobianDict = dict[str, "UACJacobianDict" | np.ndarray]


# ==================================================================================================
class UACMapper:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        original_mesh_vertices: np.ndarray,
        uac_submesh_data: constructor.UACSubmeshDict,
    ):
        self._uac_submesh_data = uac_submesh_data
        self._uac_submesh_scalar_interpolators, self._uac_submesh_vector_interpolators = (
            self._create_interpolators(uac_submesh_data)
        )
        self._uac_submesh_jacobians, self._uac_submesh_pinv_jacobians = (
            self._assemble_cellwise_jacobians(original_mesh_vertices, uac_submesh_data)
        )

    # ----------------------------------------------------------------------------------------------
    def map_scalar_from_original_to_uac_mesh(self, scalar_field: np.ndarray) -> UACScalarDict:
        uac_scalar_data = dict_utils.create_empty_dict_from_keys(self._uac_submesh_data)
        key_sequence = dict_utils.nested_dict_keys(self._uac_submesh_data)
        for key in key_sequence:
            submesh_data = dict_utils.get_dict_entry(key, self._uac_submesh_data)
            cell_index_mask = submesh_data.cell_inds
            scalar_field_on_submesh = scalar_field[cell_index_mask]
            dict_utils.set_dict_entry(key, uac_scalar_data, scalar_field_on_submesh)
        return uac_scalar_data

    # ----------------------------------------------------------------------------------------------
    def map_vector_from_original_to_uac_mesh(self, vector_field: np.ndarray) -> UACVectorDict:
        uac_vector_data = dict_utils.create_empty_dict_from_keys(self._uac_submesh_data)
        key_sequence = dict_utils.nested_dict_keys(self._uac_submesh_data)
        for key in key_sequence:
            pinv_jacobian = dict_utils.get_dict_entry(key, self._uac_submesh_pinv_jacobians)
            submesh_data = dict_utils.get_dict_entry(key, self._uac_submesh_data)
            cell_index_mask = submesh_data.cell_inds
            vector_field_on_submesh = vector_field[cell_index_mask]
            uac_vector_field = pinv_jacobian @ vector_field_on_submesh
            dict_utils.set_dict_entry(key, uac_vector_data, uac_vector_field)
        return uac_vector_data

    # ----------------------------------------------------------------------------------------------
    def map_scalar_from_uac_to_original_mesh(self, uac_scalar_data: UACScalarDict) -> np.ndarray:
        original_mesh_data = np.zeros(self._orig_mesh_simplices.shape[0])
        key_sequence = dict_utils.nested_dict_keys(self._uac_submesh_data)
        for key in key_sequence:
            submesh_data = dict_utils.get_dict_entry(key, self._uac_submesh_data)
            cell_index_mask = submesh_data.cell_inds
            uac_scalar_field_on_submesh = dict_utils.get_dict_entry(key, uac_scalar_data)
            original_mesh_data[cell_index_mask] = uac_scalar_field_on_submesh
        return original_mesh_data

    # ----------------------------------------------------------------------------------------------
    def map_vector_from_uac_to_original_mesh(self, uac_vector_data: UACVectorDict) -> np.ndarray:
        original_mesh_data = np.zeros((self._orig_mesh_simplices.shape[0], 3))
        key_sequence = dict_utils.nested_dict_keys(self._uac_submesh_data)
        for key in key_sequence:
            jacobian = dict_utils.get_dict_entry(key, self._uac_submesh_jacobians)
            submesh_data = dict_utils.get_dict_entry(key, self._uac_submesh_data)
            cell_index_mask = submesh_data.cell_inds
            vector_field_on_submesh = dict_utils.get_dict_entry(key, uac_vector_data)
            original_mesh_data[cell_index_mask, :] = jacobian @ vector_field_on_submesh
        return original_mesh_data

    # ----------------------------------------------------------------------------------------------
    def interpolate_scalar_to_other_uac_mesh(
        self,
        uac_scalar_data: UACScalarDict,
        other_uac_mesh_data: constructor.UACSubmeshDict,
    ) -> UACScalarDict:
        other_uac_scalar_data = dict_utils.create_empty_dict_from_keys(self._uac_submesh_data)
        key_sequence = dict_utils.nested_dict_keys(self._uac_submesh_data)
        for key in key_sequence:
            scalar_submesh_data = dict_utils.get_dict_entry(key, uac_scalar_data)
            other_uac_submesh = dict_utils.get_dict_entry(key, other_uac_mesh_data)
            other_uac_coords = np.hstack(
                [other_uac_submesh.alpha[:, None], other_uac_submesh.beta[:, None]]
            )
            interpolator = dict_utils.get_dict_entry(key, self._uac_submesh_scalar_interpolators)
            interpolator.values = scalar_submesh_data
            dict_utils.set_dict_entry(key, other_uac_scalar_data, interpolator(other_uac_coords))
        return other_uac_scalar_data

    # ----------------------------------------------------------------------------------------------
    def interpolate_vector_to_other_uac_mesh(
        self,
        uac_vector_data: UACVectorDict,
        other_uac_mesh_data: constructor.UACSubmeshDict,
    ) -> UACVectorDict:
        other_uac_vector_data = dict_utils.create_empty_dict_from_keys(self._uac_submesh_data)
        key_sequence = dict_utils.nested_dict_keys(self._uac_submesh_data)
        for key in key_sequence:
            vector_submesh_data = dict_utils.get_dict_entry(key, uac_vector_data)
            other_uac_submesh = dict_utils.get_dict_entry(key, other_uac_mesh_data)
            other_uac_coords = np.hstack(
                [other_uac_submesh.alpha[:, None], other_uac_submesh.beta[:, None]]
            )
            interpolator = dict_utils.get_dict_entry(key, self._uac_submesh_vector_interpolators)
            interpolator.values = vector_submesh_data
            dict_utils.set_dict_entry(key, other_uac_vector_data, interpolator(other_uac_coords))
        return other_uac_vector_data

    # ----------------------------------------------------------------------------------------------
    def _create_interpolators(self, uac_submesh_data: constructor.UACSubmeshDict) -> None:
        scalar_interpolators = dict_utils.create_empty_dict_from_keys(uac_submesh_data)
        vector_interpolators = dict_utils.create_empty_dict_from_keys(uac_submesh_data)
        key_sequence = dict_utils.nested_dict_keys(uac_submesh_data)
        for key in key_sequence:
            num_vertices = dict_utils.get_dict_entry(key, uac_submesh_data).vertex_inds.shape[0]
            dummy_scalar_values = np.zeros(num_vertices)
            dummy_vector_values = np.zeros((num_vertices, 2))
            submesh_data = dict_utils.get_dict_entry(key, uac_submesh_data)
            coords = np.hstack([submesh_data.alpha[:, None], submesh_data.beta[:, None]])
            dict_utils.set_dict_entry(
                key, scalar_interpolators, sps.NearestNDInterpolator(coords, dummy_scalar_values)
            )
            dict_utils.set_dict_entry(
                key, vector_interpolators, sps.NearestNDInterpolator(coords, dummy_vector_values)
            )
        return scalar_interpolators, vector_interpolators

    # ----------------------------------------------------------------------------------------------
    def _assemble_cellwise_jacobians(
        self, original_mesh_vertices: np.ndarray, uac_submesh_data: constructor.UACSubmeshDict
    ):
        cellwise_jacobians = dict_utils.create_empty_dict_from_keys(self._uac_submesh_data)
        pinv_cellwise_jacobians = dict_utils.create_empty_dict_from_keys(self._uac_submesh_data)
        key_sequence = dict_utils.nested_dict_keys(self._uac_submesh_data)
        for key in key_sequence:
            submesh_data = dict_utils.get_dict_entry(key, uac_submesh_data)
            vertex_inds = submesh_data.vertex_inds
            simplices = submesh_data.connectivity
            vertices_original = original_mesh_vertices[vertex_inds]
            vertices_uac = np.hstack([submesh_data.alpha[:, None], submesh_data.beta[:, None]])
            jacobians = jacobian.compute_cellwise_jacobians(
                vertices_original, vertices_uac, simplices
            )
            pinv_jacobians = np.linalg.pinv(jacobians)
            dict_utils.set_dict_entry(key, cellwise_jacobians, jacobians)
            dict_utils.set_dict_entry(key, pinv_cellwise_jacobians, pinv_jacobians)
        return cellwise_jacobians, pinv_cellwise_jacobians
