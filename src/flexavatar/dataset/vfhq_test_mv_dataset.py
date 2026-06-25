from typing import List

from flexavatar.config.dataset_config import DatasetType, SampleMetadata
from flexavatar.config.expression_config import ExpressionCodeConfig
from flexavatar.constants import VFHQ_TEST_DATASET_NAME
from flexavatar.data_adapter.pixel3dmm_data_adapter import Pixel3DMMDataAdapter
from flexavatar.data_adapter.vfhq_data_adapter import VFHQTestDataFolder, VFHQTestDataAdapter
from flexavatar.dataset.pixel3dmm_mv_dataset import Pixel3DMMMVDataset



class VFHQTestMVDataset(Pixel3DMMMVDataset):

    def _get_data_folder(self):
        return VFHQTestDataFolder()

    def _get_data_adapter(self, sample_metadata: SampleMetadata, expression_code_config: ExpressionCodeConfig) -> Pixel3DMMDataAdapter:
        return VFHQTestDataAdapter(sample_metadata.participant_id, expression_code_config)

    def _get_dataset_name(self) -> DatasetType:
        return VFHQ_TEST_DATASET_NAME

    def _list_participant_ids(self) -> List[str]:
        data_folder = self._get_data_folder()
        video_keys = data_folder.list_video_keys()
        return video_keys

if __name__ == '__main__':
    from flexavatar.model_manager.flexavatar_model_manager import FlexAvatarModelManager
    model_name = 'FLEX-1'
    model_manager = FlexAvatarModelManager(model_name)
    dataset_config = model_manager.load_dataset_config()
    dataset_config = dataset_config.make_vfhq_test_eval()
    dataset = VFHQTestMVDataset(dataset_config)
    sample = dataset[0]
    print(len(dataset))