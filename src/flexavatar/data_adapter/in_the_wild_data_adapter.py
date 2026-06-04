import math
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
from PIL import Image
from dreifus.camera import PoseType
from dreifus.matrix import Pose, Intrinsics
from dreifus.vector import Vec3
from elias.util import load_img
from elias.util.io import resize_img

from flexavatar.config.dataset_config import SampleMetadata
from flexavatar.data_adapter.pixel3dmm_data_adapter import Pixel3DMMDataAdapter
from flexavatar.env import FLEXAVATAR_INPUTS_PATH, FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH

ITW_ENLARGE_FACTOR = 0.1

class InTheWildDataAdapter(Pixel3DMMDataAdapter):
    def load_image(self, sample_metadata: SampleMetadata) -> np.ndarray:
        image = load_img(self.get_image_path(sample_metadata.participant_id))[..., :3]
        crop_params = np.load(self._get_crop_params_path())
        offset = int((crop_params[1] - crop_params[0]) * ITW_ENLARGE_FACTOR)
        # extend_bottom = max(0, offset - (image.shape[0] - crop_params[1] - 1))
        extend_bottom = max(0, offset - (image.shape[0] - crop_params[1]))
        offset_top = offset + extend_bottom
        offset_bottom = 2 * offset - offset_top
        extend_top = max(0, offset_top - crop_params[0])
        extend_left = max(0, offset - crop_params[2])
        extend_right = max(0, offset - (image.shape[1] - crop_params[3]))

        # extend_top += extend_bottom
        # extend_bottom = 0
        extend_bottom = 0

        image = np.concatenate([np.ones((extend_top, image.shape[1], 3), dtype=np.uint8) * 255, image, np.ones((extend_bottom, image.shape[1], 3), dtype=np.uint8) * 255], axis=0)
        image = np.concatenate([np.ones((image.shape[0], extend_left, 3), dtype=np.uint8) * 255, image, np.ones((image.shape[0], extend_right, 3), dtype=np.uint8) * 255], axis=1)
        # image = image[crop_params[0] - offset + extend_top: crop_params[1] + offset + extend_top, crop_params[2] - offset + extend_left:crop_params[3] + offset + extend_left]
        image = image[crop_params[0] - offset_top + extend_top: crop_params[1] + offset_bottom + extend_top, crop_params[2] - offset + extend_left:crop_params[3] + offset + extend_left]

        # image = image[crop_params[0] : crop_params[1], crop_params[2]:crop_params[3]]
        if image.shape[0] != 512 or image.shape[1] != 512:
            image = resize_img(image, (512 / image.shape[1], 512 / image.shape[0]))
        return image

    def load_head_pose(self, sample_metadata: SampleMetadata) -> Tuple[Pose, float]:
        if sample_metadata.serial in self.list_cameras_eval(sample_metadata):
            return Pose(), 1
        else:
            return super().load_head_pose(sample_metadata)

    def load_camera_params(self, sample_metadata: SampleMetadata) -> Tuple[Pose, Intrinsics]:
        if sample_metadata.serial in self.list_cameras_eval(sample_metadata):
            z_offset = -0.05
            frontal_pose = Pose(pose_type=PoseType.CAM_2_WORLD)
            degrees = int(sample_metadata.serial) / 360 * 2 * np.pi
            frontal_pose.set_translation(Vec3(math.sin(degrees), 0, math.cos(degrees) + z_offset))
            frontal_pose.look_at(Vec3(0, 0, z_offset), up=Vec3(0, 1, 0))
            pose = frontal_pose

            resolution = 512
            intrinsics = Intrinsics(1500 * resolution / 512, 1500 * resolution / 512, resolution / 2, resolution / 2)
        else:
            pose, intrinsics = super().load_camera_params(sample_metadata)

            image_height = Image.open(self.get_image_path(sample_metadata.participant_id)).size[1]
            crop_params = np.load(self._get_crop_params_path())

            offset = int((crop_params[1] - crop_params[0]) * ITW_ENLARGE_FACTOR)
            extend_bottom = max(0, offset - (image_height - crop_params[1]))
            offset_top = offset + extend_bottom
            offset_top = offset_top / (crop_params[1] - crop_params[0]) * 512

            offset = 512 * ITW_ENLARGE_FACTOR

            intrinsics = intrinsics.crop(-offset, -offset_top, inplace=False)
            intrinsics = intrinsics.rescale(512 / (512 + 2 * offset), inplace=False)

        return pose, intrinsics

    def load_mask(self, sample_metadata: SampleMetadata) -> np.ndarray:
        return np.ones((512, 512), dtype=np.uint8) * 255

    def list_cameras(self, sample_metadata: SampleMetadata) -> List[Union[str, int]]:
        return [0]

    def list_cameras_left(self, sample_metadata: SampleMetadata) -> List[Union[str, int]]:
        raise NotImplementedError()

    def list_cameras_right(self, sample_metadata: SampleMetadata) -> List[Union[str, int]]:
        raise NotImplementedError()

    def list_cameras_eval(self, sample_metadata: SampleMetadata) -> List[Union[str, int]]:
        return ["0", "90", "180", "270"]

    def list_timesteps(self, sample_metadata: SampleMetadata = None) -> List[int]:
        return [0]

    def apply_color_correction(self, image: np.ndarray, camera: Union[str, int]) -> np.ndarray:
        return image

    def get_image_path(self, image_name: str) -> str:
        image_path = f"{FLEXAVATAR_INPUTS_PATH}/itw/{image_name}.png"
        if not Path(image_path).exists():
            image_path = f"{FLEXAVATAR_INPUTS_PATH}/itw/{image_name}.jpg"

        return image_path

    def _get_crop_params_path(self) -> str:
        return f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/processing/itw/{self._video_key}/tracking/crop_ymin_ymax_xmin_xmax.npy"

    def _get_tracking_folder(self) -> str:
        return f"{self._video_key}/tracking_nV1_noPho_uv2000.0_n1000.0"

    def _get_data_folder(self) -> str:
        pass

    @classmethod
    def _get_tracking_base_path(cls) -> str:
        return f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/tracking/itw"

    @classmethod
    def _get_data_base_path(cls) -> str:
        return ""



