from dataclasses import dataclass, fields, field, replace
from enum import IntEnum
from typing import List, Union, Tuple, Optional, Literal, Dict

import torch
from dreifus.graphics import Dimensions
from dreifus.matrix import Pose, Intrinsics
from elias.config import Config, better_replace

from flexavatar.config.expression_config import ExpressionCodeConfig

TargetViewSampling = Literal['random', 'input_plus_random', 'sequential', 'right', 'same_as_input']
InputViewSampling = Literal['random', 'frontal', 'all', 'left']
TargetTimestepSampling = Literal['same_as_input', 'random', 'random_per_view', 'evenly_spaced', 'middle', 'all']
InputTimestepSampling = Literal['first_frame', 'all', 'random', '10_frames', 'evenly_spaced']


@dataclass
class MVDatasetConfig(Config):
    use_dino: bool = True
    dino_resolution: int = 224  # Input resolution to DINO

    n_participants: int = 1
    percentage_3d_participants: Optional[float] = None
    input_resolution: int = 512
    target_resolution: int = 512
    idx_participant_start: int = 0
    n_target_views: int = 1
    n_target_timesteps: int = 1
    n_input_views: int = 1
    n_input_views_condition: int = 0  # Number of clean conditioning views. Only for lifting
    mask_input_image: bool = False
    target_view_sampling: TargetViewSampling = 'input_plus_random'
    input_view_sampling: InputViewSampling = 'frontal'
    target_timestep_sampling: TargetTimestepSampling = 'same_as_input'
    input_timestep_sampling: InputTimestepSampling = 'first_frame'
    back_head_sample_weight: Optional[float] = None
    use_dinov3: bool = False  # remove
    dino_name: str = 'dinov2_vitg14_reg'
    load_precomputed_dino: bool = False
    load_expression_codes: bool = False
    expression_code_config: ExpressionCodeConfig = ExpressionCodeConfig()
    load_input_expression_codes: bool = False
    load_render_head_poses: bool = False

    extract_dino_layers: List[int] = field(default_factory=lambda: [-1])
    use_vae: bool = False
    slrm_encoder: Optional[str] = None  # remove
    finetune_slrm_decoder: bool = False
    finetune_slrm_encoder: bool = False
    vae_type: str = "ema"
    normalize_images: bool = False
    seed: int = 0
    use_random_target_cropping: bool = False
    use_random_input_cropping: bool = False
    use_custom_input_cropping: bool = False
    use_custom_target_cropping: bool = False
    min_input_crop_size: int = 512
    use_square_crops: bool = False
    use_caching: bool = True
    use_image_augmentations: bool = False
    use_random_bg_color: bool = False
    bg_color: Optional[Tuple[int, int, int]] = None
    augmentation_datasets: List[str] = field(default_factory=lambda: ['nersemble', 'cafca', 'ava256'])
    use_dataset_ids: bool = False
    use_separate_dataset_ids: bool = False
    use_nersemble_dataset_ids: bool = False  # 17 embeddings: 16 for nersemble cameras + 1 for everything else
    prob_nvs_task: float = 0  # Probability that a sample will be for an NVS task (input timestep = output timestep, expr codes = 0s)
    temp_jaw_angle: float = 0  # Temperature for sampling more open mouths. 0 == uniform sampling, 1 == prob of timestep is FLAME jaw angle

    use_cafca: bool = False  # Only for logging
    use_celebvtext: bool = False  # Only for logging
    use_hello3: bool = False
    use_celebvhq: bool = False
    use_nersemble: bool = True
    use_ava256: bool = False
    use_hold_out_ids: bool = False  # Rather naive way to get unseen identities
    use_hold_out_ids2: bool = False  # Rather naive way to get unseen identities
    # NeRSemble
    apply_color_correction: bool = False
    filter_bad_videos: bool = False

    # NeRSemble benchmark
    n_input_sequences: int = 1

    # VFHQ
    use_cross_reenactment: bool = False

    # Ava256
    exclude_avat3r_participants: bool = False

    # Dataset weights
    nersemble_weight: float = 1
    ava256_weight: float = 1

    @classmethod
    def _backward_compatibility(cls, loaded_config: Dict):
        if 'expression_code_type' in loaded_config:
            # expression_code_type was moved into expression_code_config
            loaded_config['expression_code_config'] = ExpressionCodeConfig(expression_code_type=loaded_config['expression_code_type'])
            del loaded_config['expression_code_type']
        super()._backward_compatibility(loaded_config)

    def make_inference(self):
        return better_replace(self,
                              use_random_target_cropping=False,
                              use_random_input_cropping=False,
                              use_random_bg_color=False,
                              use_image_augmentations=False,
                              prob_nvs_task=0,
                              input_view_sampling='frontal',
                              input_timestep_sampling='first_frame')

    def make_in_the_wild_eval(self) -> 'MVDatasetConfig':
        return better_replace(self,
                              n_target_timesteps=1,
                              target_view_sampling='eval',
                              input_view_sampling='frontal',
                              input_timestep_sampling='first_frame',
                              target_timestep_sampling='same_as_input',
                              use_random_target_cropping=False,
                              use_random_input_cropping=False,
                              use_square_crops=True,
                              use_random_bg_color=False,
                              use_image_augmentations=False,
                              apply_color_correction=True,
                              prob_nvs_task=0,
                              percentage_3d_participants=None)



DatasetType = Literal[
    'nersemble', 'cafca', 'celebvtext', 'hello3', 'in_the_wild', 'phone_scan', 'ava256', 'ava256_avat3r', 'vfhq_test', 'vfhq_test_50', 'nersemble_benchmark', 'ava256_static', 'ava256_static_back']
DATASET_ID_MAPPING = ['nersemble', 'cafca', 'ava256', 'celebvtext', 'hello3', 'in_the_wild', 'phone_scan', 'vfhq_test', 'vfhq_test_50', 'ava256_avat3r',
                      'nersemble_benchmark']


@dataclass(order=True)
class SampleMetadata:
    participant_id: Union[str, int]
    sequence_name: Union[str, int]
    timestep: Union[str, int]
    serial: Union[str, int]
    dataset: DatasetType = 'nersemble'
    environment: int = 0

    def __hash__(self) -> int:
        return hash((self.dataset, self.participant_id, self.sequence_name, self.timestep, self.serial, self.environment))
        # return hash(f"{self.dataset}_{self.participant_id}_{self.sequence_name}_{self.timestep}_{self.serial}")


@dataclass
class MVDatasetInferenceSample:
    input_images: torch.Tensor  # [VI, 3, H, W]
    input_sample_metadatas: List[SampleMetadata]  # [VI]
    input_cam2worlds: List[Pose]  # [VI]
    input_intrinsics: List[Intrinsics]  # [VI]
    input_view_mask: torch.Tensor  # [VI]
    input_expression_codes: Optional[torch.Tensor]  # [VI]
    input_masks: Optional[torch.Tensor]  # [VI, H, W]  # Not needed for inference
    dataset_ids: Optional[torch.Tensor]  # [1]


@dataclass
class MVDatasetSample(MVDatasetInferenceSample):
    render_cam2world_poses: List[Pose]
    render_intrinsics: List[Intrinsics]
    render_resolution: Union[Dimensions, int]
    render_bg_color: Tuple[int, int, int]
    target_images: torch.Tensor  # [VT, 3, H, W]
    target_sample_metadatas: Optional[List[SampleMetadata]] = None  # [VT]
    expression_codes: Optional[torch.Tensor] = None  # [VT, D_exp]
    render_head_poses: Optional[List[Pose]] = None

    @staticmethod
    def collate_inputs(samples: List['MVDatasetSample']) -> 'MVDatasetSample':
        tensor_fields = ["input_images", "input_view_mask", "input_expression_codes", "input_masks"]
        list_fields = ["input_sample_metadatas", "input_cam2worlds", "input_intrinsics"]
        collated_values = dict()
        for field in tensor_fields:
            values = [getattr(sample, field) for sample in samples]
            if values[0] is None:
                collated_value = None
            else:
                collated_value = torch.cat(values, dim=0)
            collated_values[field] = collated_value

        for field in list_fields:
            values = [getattr(sample, field) for sample in samples]
            if values[0] is None:
                collated_value = None
            else:
                collated_value = [item for items in values for item in items]
            collated_values[field] = collated_value

        collated_sample = replace(samples[0], **collated_values)
        return collated_sample


@dataclass
class FlexAvatarBatch:
    input_images: torch.Tensor  # [B, VI, C, H, W]
    input_sample_metadatas: List[List[SampleMetadata]]  # [B, VI]
    input_cam2worlds: List[List[Pose]]  # [B, VI]
    input_intrinsics: List[List[Intrinsics]]  # [B, VI]
    input_view_mask: torch.Tensor  # [B, VI]  # 0: noisy input, 1: clean input
    render_cam2world_poses: List[List[Pose]]  # [B, VT]
    render_intrinsics: List[List[Intrinsics]]  # [B, VT]
    render_resolution: List[Union[Dimensions, int]]  # [VT]
    render_bg_color: List[Tuple[int, int, int]]  # [VT]

    target_images: torch.Tensor  # [B, VT, C, H, W]
    target_sample_metadatas: Optional[List[List[SampleMetadata]]] = None  # [B, VT]
    features: Optional[torch.Tensor] = None  # [B, VI, C_f, H_f, W_f]
    expression_codes: Optional[torch.Tensor] = None  # [B, VT, D_exp]
    residual_codes: Optional[torch.Tensor] = None  # [B, VT, D_res]
    input_expression_codes: Optional[torch.Tensor] = None  # [B, VI, D_exp]
    input_masks: Optional[torch.Tensor] = None  # [B, VI, H, W]
    render_view_mask: Optional[torch.Tensor] = None  # [B, VT]
    render_head_poses: Optional[List[List[Pose]]] = None  # [B, VT]
    dataset_ids: Optional[torch.Tensor] = None  # [B]

    def __getitem__(self, item) -> 'FlexAvatarBatch':
        selected_values = dict()
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                if isinstance(value, list):
                    value = [value[i] for i in item]
                else:
                    value = value[item]
            selected_values[field.name] = value

        return replace(self, **selected_values)

    @property
    def B(self) -> int:
        return self.input_images.shape[0]

    @property
    def VI(self) -> int:
        return self.input_images.shape[1]

    @property
    def VT(self) -> int:
        return len(self.render_cam2world_poses[0])

    @property
    def device(self) -> torch.device:
        return self.input_images.device

    def __len__(self) -> int:
        return self.input_images.shape[0]

    def to(self, device: torch.device):
        for f in fields(self):
            value = getattr(self, f.name)
            value = self._recursive_to(value, device)
            setattr(self, f.name, value)

        return self

    def _recursive_to(self, value, device: torch.device) -> object:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        elif isinstance(value, list):
            return [self._recursive_to(item, device) for item in value]

        return value

    def pin_memory(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                value = value.pin_memory()
                setattr(self, f.name, value)
        return self
