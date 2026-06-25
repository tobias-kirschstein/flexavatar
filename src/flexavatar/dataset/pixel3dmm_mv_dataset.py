from abc import abstractmethod
from typing import List

import numpy as np

from flexavatar.config.dataset_config import DatasetType, SampleMetadata
from flexavatar.config.expression_config import ExpressionCodeConfig
from flexavatar.data_adapter.pixel3dmm_data_adapter import Pixel3DMMDataAdapter
from flexavatar.dataset.base_mv_dataset import BaseMVDataset
from flexavatar.util.crop import Crop


class Pixel3DMMMVDataset(BaseMVDataset):

    @abstractmethod
    def _get_data_folder(self):
        pass

    @abstractmethod
    def _get_data_adapter(self, sample_metadata: SampleMetadata, expression_code_config: ExpressionCodeConfig) -> Pixel3DMMDataAdapter:
        pass

    @abstractmethod
    def _get_dataset_name(self) -> DatasetType:
        pass

    def _is_3d_dataset(self) -> bool:
        return False

    def _list_participant_ids(self) -> List[int]:
        data_folder = self._get_data_folder()
        if self._config.filter_bad_videos:
            video_keys = data_folder.list_valid_video_keys()
        else:
            video_keys = data_folder.list_video_keys()

        return video_keys

    def _list_sample_metadatas(self) -> List[SampleMetadata]:
        camera = 0

        sample_metadatas = []
        for participant_id in self._participant_ids:
            data_adapter = self._get_data_adapter(SampleMetadata(participant_id, None, None, None), ExpressionCodeConfig())

            if self._config.input_timestep_sampling == 'first_frame':
                timesteps = [0]
            elif self._config.input_timestep_sampling == 'random':
                timesteps = ['random']
            elif self._config.input_timestep_sampling == '10_frames':
                timesteps = data_adapter.list_timesteps()
                timesteps = [timesteps[idx] for idx in np.linspace(0, len(timesteps) - 1, min(10, len(timesteps)), dtype=int)]
            else:
                timesteps = data_adapter.list_timesteps()

            for t in timesteps:
                sample_metadatas.append(SampleMetadata(participant_id, None, t, camera, dataset=self._get_dataset_name()))

        return sample_metadatas

    def _get_fixed_head_crop(self) -> Crop:
        return Crop(0, 0, 512, 512)