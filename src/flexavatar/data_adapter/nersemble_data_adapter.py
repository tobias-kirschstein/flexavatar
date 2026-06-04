from typing import List, Union

from flexavatar.config.dataset_config import SampleMetadata
from flexavatar.config.expression_config import ExpressionCodeConfig
from flexavatar.data_adapter.pixel3dmm_data_adapter import Pixel3DMMDataAdapter
from flexavatar.env import FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH


class NeRSembleDataAdapter(Pixel3DMMDataAdapter):
    def __init__(self, participant_id: int, sequence_name: str, expression_code_config: ExpressionCodeConfig = ExpressionCodeConfig()):
        super().__init__(None, expression_code_config)
        self._participant_id = participant_id
        self._sequence_name = sequence_name

    def list_cameras_left(self, sample_metadata: SampleMetadata) -> List[Union[str, int]]:
        pass

    def list_cameras_right(self, sample_metadata: SampleMetadata) -> List[Union[str, int]]:
        pass

    @classmethod
    def _get_tracking_base_path(cls) -> str:
        return f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/tracking/nersemble"

    @classmethod
    def _get_data_base_path(cls) -> str:
        pass

    def _get_tracking_folder(self) -> str:
        return f"{self._participant_id:03d}/{self._sequence_name}/222200037/tracking_nV1_noPho_uv2000.0_n1000.0"

    def _get_data_folder(self) -> str:
        pass