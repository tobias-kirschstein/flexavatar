import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import mediapy
import numpy as np
import torch
from elias.config import Config
from elias.folder import Folder
from elias.util import load_img, save_json, load_json, save_img, ensure_directory_exists_for_file
from torch.nn.functional import upsample_nearest
from visage.evaluator.paired_face_image_evaluator import PairedFaceImageMetrics

from flexavatar.config.dataset_config import DatasetType, SampleMetadata
from flexavatar.constants import VFHQ_TEST_DATASET_NAME
from flexavatar.env import FLEXAVATAR_EVALUATIONS_PATH


@dataclass
class EvaluationResult(Config):
    face_image_metrics: PairedFaceImageMetrics


@dataclass
class EvaluationConfig(Config):
    run_fitting: bool = False
    n_fitting_steps: int = 200
    crop_vfhq: bool = False
    black: bool = False
    cross_reenactment: bool = False


class EvaluationManager:

    def __init__(self, run_name: str, dataset_type: DatasetType = VFHQ_TEST_DATASET_NAME, evaluation_config: EvaluationConfig = EvaluationConfig()):
        evaluation_name = ""
        if evaluation_config.run_fitting:
            evaluation_name += f"_inv-{evaluation_config.n_fitting_steps}"
        if evaluation_config.black:
            evaluation_name += f"_black"
        if evaluation_config.cross_reenactment:
            dataset_type += f"_cross-reenactment"

        if evaluation_name:
            self._evaluation_folder = f"{FLEXAVATAR_EVALUATIONS_PATH}/{run_name}{evaluation_name}"
        else:
            self._evaluation_folder = f"{FLEXAVATAR_EVALUATIONS_PATH}/{run_name}"

        self._run_name = run_name
        self._dataset_type = dataset_type
        self._evaluation_config = evaluation_config

    def list_evaluated_checkpoints(self) -> List[int]:
        checkpoints = Folder(self._evaluation_folder).list_file_numbering(f"evaluation_{self._dataset_type}_ckpt$k.json", return_only_numbering=True)
        return checkpoints

    def list_checkpoints_with_predictions(self) -> List[int]:
        checkpoints = Folder(self._evaluation_folder).list_file_numbering(f"images_{self._dataset_type}_ckpt$k", return_only_numbering=True)
        return checkpoints

    def list_sample_metadatas(self, checkpoint: int) -> List[SampleMetadata]:
        sample_metadatas = list()
        pattern = re.compile("^(.+)_s(.+)_t(\d+)_c(.+)_pred\.png$")
        for image_path in Path(self.get_images_folder(checkpoint)).iterdir():
            matches = pattern.match(image_path.name)
            if matches:
                participant_id = matches.group(1)
                sequence_name = matches.group(2)
                timestep = int(matches.group(3))
                serial = matches.group(4)
                sample_metadata = SampleMetadata(participant_id, sequence_name, timestep, serial, dataset=self._dataset_type)
                sample_metadatas.append(sample_metadata)

        sample_metadatas = sorted(sample_metadatas)
        return sample_metadatas

    def save_prediction_image(self, image: np.ndarray, sample_metadata: SampleMetadata, checkpoint: int):
        save_img(image, self.get_prediction_image_path(sample_metadata, checkpoint))

    def load_prediction_image(self, sample_metadata: SampleMetadata, checkpoint: int) -> np.ndarray:
        return load_img(self.get_prediction_image_path(sample_metadata, checkpoint))

    def save_target_image(self, image: np.ndarray, sample_metadata: SampleMetadata, checkpoint: int):
        save_img(image, self.get_target_image_path(sample_metadata, checkpoint))

    def load_target_image(self, sample_metadata: SampleMetadata, checkpoint: int) -> np.ndarray:
        return load_img(self.get_target_image_path(sample_metadata, checkpoint))

    def save_evaluation_result(self, evaluation_result: EvaluationResult, checkpoint: int):
        save_json(evaluation_result.to_json(), self.get_evaluation_result_path(checkpoint))

    def load_evaluation_result(self, checkpoint: int) -> EvaluationResult:
        return EvaluationResult.from_json(load_json(self.get_evaluation_result_path(checkpoint)))

    def save_input_image(self, input_image: np.ndarray, participant_id: str):
        save_img(input_image, self.get_input_image_path(participant_id))

    def load_input_image(self, participant_id: str) -> np.ndarray:
        return load_img(self.get_input_image_path(participant_id))

    def save_fitting_video(self, fitting_history: List[np.ndarray], participant_id: str, checkpoint: int):
        fitting_video_path = self.get_fitting_video_path(participant_id, checkpoint)
        ensure_directory_exists_for_file(fitting_video_path)
        mediapy.write_video(fitting_video_path, fitting_history)

    def save_avatar_code(self, avatar_code: torch.Tensor, participant_id: str, checkpoint: int):
        avatar_code_path = self.get_avatar_code_path(participant_id, checkpoint)
        ensure_directory_exists_for_file(avatar_code_path)
        np.save(avatar_code_path, avatar_code.cpu().numpy())

    def load_avatar_code(self, participant_id: str, checkpoint: int) -> torch.Tensor:
        avatar_code = np.load(self.get_avatar_code_path(participant_id, checkpoint))
        return torch.from_numpy(avatar_code).float()

    def save_avatar_code_image(self, avatar_code: torch.Tensor, participant_id: str, checkpoint: int, avatar_code_scale_factor: float = 0.5):
        res_uv = int(np.sqrt(avatar_code.shape[0]).item())
        avatar_code_img = avatar_code.reshape(res_uv, res_uv, avatar_code.shape[-1])
        avatar_code_img = upsample_nearest(avatar_code_img.permute(2, 0, 1)[None], 512)[0].permute(1, 2, 0)
        avatar_code_img = np.clip(255 * avatar_code_img.float().cpu().numpy()[..., :3] * avatar_code_scale_factor, 0, 255).astype(np.uint8)
        save_img(avatar_code_img, self.get_avatar_code_image_path(participant_id, checkpoint))

    # ----------------------------------------------------------
    # Paths
    # ----------------------------------------------------------

    def get_images_folder(self, checkpoint: int) -> str:
        return f"{self._evaluation_folder}/images_{self._dataset_type}_ckpt{checkpoint}k"

    def get_videos_folder(self, checkpoint: int) -> str:
        return f"{self._evaluation_folder}/videos_{self._dataset_type}_ckpt{checkpoint}k"

    def get_prediction_video_path(self, participant_id, checkpoint: int):
        video_name = f"{participant_id}"
        video_name += "_pred"

        return f"{self.get_videos_folder(checkpoint)}/{video_name}.mp4"

    def get_prediction_image_path(self, sample_metadata: SampleMetadata, checkpoint: int) -> str:
        image_name = self.get_image_base_name(sample_metadata)
        image_name += "_pred"

        return f"{self.get_images_folder(checkpoint)}/{image_name}.png"

    def get_target_image_path(self, sample_metadata: SampleMetadata, checkpoint: int) -> str:
        return f"{self.get_images_folder(checkpoint)}/{self.get_image_base_name(sample_metadata)}_target.png"

    def get_image_base_name(self, sample_metadata: SampleMetadata) -> str:
        return f"{sample_metadata.participant_id}_s{sample_metadata.sequence_name}_t{sample_metadata.timestep}_c{sample_metadata.serial}"

    def get_evaluation_result_path(self, checkpoint: int) -> str:
        evaluation_result_name = f"evaluation_{self._dataset_type}_ckpt{checkpoint}k"
        if self._evaluation_config.crop_vfhq:
            evaluation_result_name += "_crop-vfhq"
        return f"{self._evaluation_folder}/{evaluation_result_name}.json"

    def get_input_image_path(self, participant_id: str) -> str:
        return f"{self._evaluation_folder}/input_images_{self._dataset_type}/{participant_id}.png"

    def get_fitting_video_path(self, participant_id: str, checkpoint: int) -> str:
        return f"{self._evaluation_folder}/fitting_{self._dataset_type}_ckpt{checkpoint}k/{participant_id}.mp4"

    def get_avatar_code_path(self, participant_id: str, checkpoint: int) -> str:
        return f"{self._evaluation_folder}/codes_{self._dataset_type}_ckpt{checkpoint}k/avatar_code_{participant_id}.npy"

    def get_avatar_code_image_path(self, participant_id: str, checkpoint: int) -> str:
        return f"{self._evaluation_folder}/code_images_{self._dataset_type}_ckpt{checkpoint}k/avatar_code_{participant_id}.png"
