from typing import Union, Optional

import torch
from elias.manager import ModelManager
from elias.manager.model import _OptimizationConfigType

from flexavatar.config.dataset_config import MVDatasetConfig
from flexavatar.config.flexavatar_config import FlexAvatarModelConfig
from flexavatar.env import FLEXAVATAR_MODELS_PATH
from flexavatar.model.flexavatar_model import FlexAvatarModel


class FlexAvatarModelManager(ModelManager[FlexAvatarModel, FlexAvatarModelConfig, None, MVDatasetConfig, None, None, None]):
    def __init__(self, model_name: str):
        super().__init__(FLEXAVATAR_MODELS_PATH, model_name, checkpoint_name_format="ckpt-$k.pt", checkpoints_sub_folder="checkpoints")

    def _build_model(self, model_config: FlexAvatarModelConfig, optimization_config: Optional[_OptimizationConfigType] = None, **kwargs) -> FlexAvatarModel:
        return FlexAvatarModel(model_config)

    def _store_checkpoint(self, model: FlexAvatarModel, checkpoint_file_name: str, **kwargs):
        pass

    def _load_checkpoint(self, checkpoint_file_name: Union[str, int], **kwargs) -> FlexAvatarModel:
        model_config = self.load_model_config()
        model_config.use_bfloat16 = False
        model = self.build_model(model_config)
        self._load_checkpoint_into_model(checkpoint_file_name, model)

        return model

    def _load_checkpoint_into_model(self, checkpoint_file_name: Union[str, int], model: FlexAvatarModel):
        checkpoint_file_path = f"{self._checkpoints_folder}/{checkpoint_file_name}"

        state_dict = torch.load(checkpoint_file_path)
        model.load_state_dict(state_dict)