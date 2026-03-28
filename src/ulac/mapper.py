from typing import Self

import numpy as np
import pyvista as pv

from ulac.common import dict_utils
from ulac.construction import constructor

from .common import jacobian

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
        original_mesh_num_simplices: int,
        uac_submesh_data: constructor.UACSubmeshDict,
    ):
        self.uac_pv_submeshes = self._create_pv_submeshes(uac_submesh_data)
        self.uac_submesh_data = uac_submesh_data
        self.uac_submesh_jacobians, self.uac_submesh_pinv_jacobians = (
            self._assemble_cellwise_jacobians(original_mesh_vertices, uac_submesh_data)
        )
        self._orig_mesh_num_simplices = original_mesh_num_simplices

    # ----------------------------------------------------------------------------------------------
    def map_scalar_from_original_to_uac_mesh(self, scalar_field: np.ndarray) -> UACScalarDict:
        uac_scalar_data = dict_utils.create_empty_dict_from_keys(self.uac_submesh_data)
        key_sequence = dict_utils.nested_dict_keys(self.uac_submesh_data)
        for key in key_sequence:
            submesh_data = dict_utils.get_dict_entry(key, self.uac_submesh_data)
            cell_index_mask = submesh_data.cell_inds
            scalar_field_on_submesh = scalar_field[cell_index_mask]
            dict_utils.set_dict_entry(key, uac_scalar_data, scalar_field_on_submesh)
        return uac_scalar_data

    # ----------------------------------------------------------------------------------------------
    def map_scalar_from_uac_to_original_mesh(self, uac_scalar_data: UACScalarDict) -> np.ndarray:
        original_mesh_data = np.zeros(self._orig_mesh_num_simplices)
        key_sequence = dict_utils.nested_dict_keys(self.uac_submesh_data)
        for key in key_sequence:
            submesh_data = dict_utils.get_dict_entry(key, self.uac_submesh_data)
            cell_index_mask = submesh_data.cell_inds
            uac_scalar_field_on_submesh = dict_utils.get_dict_entry(key, uac_scalar_data)
            original_mesh_data[cell_index_mask] = uac_scalar_field_on_submesh
        return original_mesh_data

    # ----------------------------------------------------------------------------------------------
    def map_vector_from_original_to_uac_mesh(self, vector_field: np.ndarray) -> UACVectorDict:
        uac_vector_data = dict_utils.create_empty_dict_from_keys(self.uac_submesh_data)
        key_sequence = dict_utils.nested_dict_keys(self.uac_submesh_data)
        for key in key_sequence:
            pinv_jacobian = dict_utils.get_dict_entry(key, self.uac_submesh_pinv_jacobians)
            submesh_data = dict_utils.get_dict_entry(key, self.uac_submesh_data)
            cell_index_mask = submesh_data.cell_inds
            vector_field_on_submesh = vector_field[cell_index_mask]
            uac_vector_field = np.einsum("ijk,ik->ij", pinv_jacobian, vector_field_on_submesh)
            dict_utils.set_dict_entry(key, uac_vector_data, uac_vector_field)
        return uac_vector_data

    # ----------------------------------------------------------------------------------------------
    def map_vector_from_uac_to_original_mesh(self, uac_vector_data: UACVectorDict) -> np.ndarray:
        original_mesh_data = np.zeros((self._orig_mesh_num_simplices, 3))
        key_sequence = dict_utils.nested_dict_keys(self.uac_submesh_data)
        for key in key_sequence:
            jacobian = dict_utils.get_dict_entry(key, self.uac_submesh_jacobians)
            submesh_data = dict_utils.get_dict_entry(key, self.uac_submesh_data)
            cell_index_mask = submesh_data.cell_inds
            vector_field_on_submesh = dict_utils.get_dict_entry(key, uac_vector_data)
            original_mesh_data[cell_index_mask, :] = np.einsum(
                "ijk,ik->ij", jacobian, vector_field_on_submesh
            )
        return original_mesh_data

    # ----------------------------------------------------------------------------------------------
    def interpolate_to_other_uac_mesh(
        self,
        uac_data: UACScalarDict | UACVectorDict,
        other_uac_mapper: Self,
    ) -> UACScalarDict | UACVectorDict:
        other_uac_data = dict_utils.create_empty_dict_from_keys(self.uac_submesh_data)
        key_sequence = dict_utils.nested_dict_keys(self.uac_submesh_data)
        for key in key_sequence:
            submesh_data = dict_utils.get_dict_entry(key, uac_data)
            pv_submesh = dict_utils.get_dict_entry(key, self.uac_pv_submeshes)
            other_pv_submesh = dict_utils.get_dict_entry(key, other_uac_mapper.uac_pv_submeshes)
            transferred_submesh_data = self._interpolate_to_other_pv_submesh(
                submesh_data, pv_submesh, other_pv_submesh
            )
            pv_submesh.clear_data()
            other_pv_submesh.clear_data()
            dict_utils.set_dict_entry(key, other_uac_data, transferred_submesh_data)
        return other_uac_data

    # ----------------------------------------------------------------------------------------------
    def _create_pv_submeshes(self, uac_submesh_data: constructor.UACSubmeshDict):
        pv_submeshes = dict_utils.create_empty_dict_from_keys(uac_submesh_data)
        key_sequence = dict_utils.nested_dict_keys(uac_submesh_data)
        for key in key_sequence:
            submesh_data = dict_utils.get_dict_entry(key, uac_submesh_data)
            vertex_coordinates = np.hstack(
                [
                    submesh_data.alpha[:, None],
                    submesh_data.beta[:, None],
                    np.zeros((submesh_data.alpha.shape[0], 1)),
                ]
            )
            simplicies = submesh_data.connectivity
            pv_submesh = pv.PolyData.from_regular_faces(vertex_coordinates, simplicies)
            dict_utils.set_dict_entry(key, pv_submeshes, pv_submesh)
        return pv_submeshes

    # ----------------------------------------------------------------------------------------------
    def _assemble_cellwise_jacobians(
        self, original_mesh_vertices: np.ndarray, uac_submesh_data: constructor.UACSubmeshDict
    ):
        cellwise_jacobians = dict_utils.create_empty_dict_from_keys(self.uac_submesh_data)
        pinv_cellwise_jacobians = dict_utils.create_empty_dict_from_keys(self.uac_submesh_data)
        key_sequence = dict_utils.nested_dict_keys(self.uac_submesh_data)
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

    # ----------------------------------------------------------------------------------------------
    def _interpolate_to_other_pv_submesh(
        self,
        scalar_submesh_data: np.ndarray,
        pv_submesh: pv.PolyData,
        other_pv_submesh: pv.PolyData,
    ) -> np.ndarray:
        pv_submesh.cell_data["scalar_to_transfer"] = scalar_submesh_data
        pv_submesh = pv_submesh.cell_data_to_point_data()
        other_pv_submesh = other_pv_submesh.sample(pv_submesh)
        other_pv_submesh = other_pv_submesh.point_data_to_cell_data()
        other_scalar_submesh_data = other_pv_submesh.cell_data["scalar_to_transfer"]
        return other_scalar_submesh_data
