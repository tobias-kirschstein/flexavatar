from pathlib import Path
from typing import Tuple, List

import numpy as np
from dreifus.matrix import Pose, Intrinsics
from elias.util import load_img

from flexavatar.config.dataset_config import SampleMetadata
from flexavatar.constants import VFHQ_TEST_DATASET_NAME
from flexavatar.data_adapter.pixel3dmm_data_adapter import Pixel3DMMDataAdapter
from flexavatar.env import FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH, FLEXAVATAR_DATASETS_PATH


class VFHQTestDataFolder:
    def __init__(self):
        self._location = f"{FLEXAVATAR_DATASETS_PATH}/{VFHQ_TEST_DATASET_NAME}"

    def list_video_keys(self) -> List[str]:
        video_keys = [path.name for path in Path(self._location).iterdir()]
        video_keys = list(sorted(video_keys))
        return video_keys


class VFHQTestDataAdapter(Pixel3DMMDataAdapter):

    @classmethod
    def _get_tracking_base_path(cls) -> str:
        return f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/tracking/{VFHQ_TEST_DATASET_NAME}/"

    @classmethod
    def _get_data_base_path(cls) -> str:
        return f"{FLEXAVATAR_DATASETS_PATH}/{VFHQ_TEST_DATASET_NAME}"

    def _get_tracking_folder(self) -> str:
        return f"{self._video_key}/tracking_nV1_noPho_uv2000.0_n1000.0"

    def _get_data_folder(self) -> str:
        return f"{self._video_key}"

    def load_image(self, sample_metadata: SampleMetadata) -> np.ndarray:
        image_path = f"{self._get_data_base_path()}/{self._get_data_folder()}/{sample_metadata.timestep:08d}.png"
        return load_img(image_path)

    def load_camera_params(self, sample_metadata: SampleMetadata) -> Tuple[Pose, Intrinsics]:
        cam2world_pose, intrinsics = super().load_camera_params(sample_metadata)
        crop_metadata_path = f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/processing/{VFHQ_TEST_DATASET_NAME}/{self._video_key}/tracking/crop_ymin_ymax_xmin_xmax.npy"
        y_min, y_max, x_min, x_max = np.load(crop_metadata_path)
        # Undo Effect of cropping on intrinsics, i.e., undo the following steps
        # intrinsics.crop(x_min, y_min)
        # intrinsics.rescale(512/ (y_max - y_min))

        intrinsics.rescale((y_max - y_min) / 512)
        intrinsics.crop(-x_min, -y_min)
        return cam2world_pose, intrinsics
