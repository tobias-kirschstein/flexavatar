from abc import abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Union, List, Tuple, Optional, Literal
from zipfile import ZipFile

import numpy as np
import torch
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose, Intrinsics
from elias.util import load_img

from flexavatar.config.dataset_config import SampleMetadata
from flexavatar.config.expression_config import ExpressionCodeConfig
from flexavatar.data_adapter.base_data_adapter import BaseDataAdapter
from flexavatar.util.rotation import rotation_6d_to_matrix

def _nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    # Project a near-orthonormal 3x3 matrix onto the closest proper rotation (det = +1).
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation.astype(matrix.dtype)


class Pixel3DMMDataAdapter(BaseDataAdapter):

    def __init__(self, video_key: str, expression_code_config: ExpressionCodeConfig = ExpressionCodeConfig()):
        self._video_key = video_key
        self._expression_code_config = expression_code_config

    def load_image(self, sample_metadata: SampleMetadata) -> np.ndarray:
        image_path = f"{self._get_data_folder()}/cropped/{sample_metadata.timestep:05d}.jpg"
        image = load_img(f"{self._get_data_base_path()}/{image_path}")

        return image

    def load_mask(self, sample_metadata: SampleMetadata) -> np.ndarray:
        mask_path = f"{PHO3DMM_MATANYONE_PATH}/{sample_metadata.dataset}/{self._video_key}/{sample_metadata.timestep:05d}.png"
        mask = load_img(mask_path)

        return mask

    def load_head_pose(self, sample_metadata: SampleMetadata) -> Tuple[Pose, float]:
        tracking = self._load_tracking(sample_metadata.timestep)

        head_rot = rotation_6d_to_matrix(torch.from_numpy(tracking['flame']['R'])).numpy()[0]

        flame2world = np.eye(4)
        flame2world[:3, :3] = head_rot
        flame2world[:3, 3] = np.squeeze(tracking['flame']['t'])
        # TODO include neck transform as well
        canonical_flame_to_world = flame2world @ tracking['joint_transforms'][0, 1, :, :]
        # The rotation block is a product of (float32) rotations, so accumulated numerical
        # error makes it fail dreifus' strict orthonormality check. Re-orthonormalize via SVD
        # (project onto the nearest valid rotation matrix) before constructing the Pose.
        canonical_flame_to_world[:3, :3] = _nearest_rotation(canonical_flame_to_world[:3, :3])
        canonical_flame_to_world = Pose(canonical_flame_to_world)

        return canonical_flame_to_world, 1

    def load_camera_params(self, sample_metadata: SampleMetadata) -> Tuple[Pose, Intrinsics]:
        tracking = self._load_tracking(sample_metadata.timestep)
        extr_open_gl_world_to_cam = np.eye(4)
        extr_open_gl_world_to_cam[:3, :3] = tracking['camera']['R_base_0'][0]
        extr_open_gl_world_to_cam[:3, 3] = tracking['camera']['t_base_0'][0]

        extr_open_gl_world_to_cam = Pose(extr_open_gl_world_to_cam,
                                         camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL,
                                         pose_type=PoseType.WORLD_2_CAM)
        cam2world_pose = extr_open_gl_world_to_cam.invert().change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False)

        image_size = 512
        intrinsics = np.eye(3)
        intrinsics[0, 0] = tracking['camera']['fl'][0, 0] * image_size
        intrinsics[1, 1] = tracking['camera']['fl'][0, 0] * image_size
        intrinsics[:2, 2] = tracking['camera']['pp'][0] * (image_size / 2 + 0.5) + image_size / 2 + 0.5

        intrinsics = Intrinsics(intrinsics)

        return cam2world_pose, intrinsics

    def load_expression_code(self, sample_metadata: SampleMetadata) -> np.ndarray:
        if self._expression_code_config.expression_code_type == 'flame':
            tracking = self._load_tracking(sample_metadata.timestep)
            expression_code = self._expression_code_config.from_pixel3dmm_tracking(tracking)

        return expression_code


    def list_cameras(self, sample_metadata: SampleMetadata) -> List[Union[str, int]]:
        return [0]

    def list_cameras_eval(self, sample_metadata: SampleMetadata) -> List[Union[str, int]]:
        pass

    def list_cameras_left(self, sample_metadata: SampleMetadata) -> List[Union[str, int]]:
        raise NotImplementedError()

    def list_cameras_right(self, sample_metadata: SampleMetadata) -> List[Union[str, int]]:
        raise NotImplementedError()

    def list_timesteps(self, sample_metadata: Optional[SampleMetadata] = None) -> List[int]:
        tracking_subfolder = f"{self._get_tracking_folder()}/checkpoint/"
        n_timesteps = len(list(Path(f"{self._get_tracking_base_path()}/{tracking_subfolder}").iterdir()))

        return list(range(n_timesteps))

    def apply_color_correction(self, image: np.ndarray, camera: Union[str, int]) -> np.ndarray:
        return image

    def _load_tracking(self, timestep: int):
        tracking_path = f"{self._get_tracking_folder()}/checkpoint/{timestep:05d}.frame"

        tracking = torch.load(f"{self._get_tracking_base_path()}/{tracking_path}", weights_only=False)
        return tracking

    @classmethod
    @abstractmethod
    def _get_tracking_base_path(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def _get_data_base_path(cls) -> str:
        pass

    @abstractmethod
    def _get_tracking_folder(self) -> str:
        pass

    @abstractmethod
    def _get_data_folder(self) -> str:
        pass