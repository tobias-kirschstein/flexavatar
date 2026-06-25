from abc import abstractmethod
from collections import defaultdict
from dataclasses import fields, replace
from math import ceil
from typing import List, Tuple, Optional

import numpy as np
import torch
from antlr4.error.Errors import IllegalStateException
from dreifus.camera import PoseType
from dreifus.graphics import Dimensions
from dreifus.matrix import Pose, Intrinsics
from elias.util.io import resize_img
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms.v2 import ColorJitter, Compose, RandomApply

from flexavatar.config.dataset_config import MVDatasetConfig, SampleMetadata, MVDatasetSample, MVDatasetInferenceSample, FlexAvatarBatch
from flexavatar.config.expression_config import ExpressionCodeConfig
from flexavatar.data_adapter.base_data_adapter import BaseDataAdapter
from flexavatar.util.crop import Crop


class BaseMVDataset(Dataset):

    def __init__(self, config: MVDatasetConfig):
        self._config = config

        participant_ids = self._list_participant_ids()
        participant_ids = participant_ids[self._config.idx_participant_start:]
        if self._config.n_participants > -1:
            participant_ids = participant_ids[:self._config.n_participants]
        if self._config.percentage_3d_participants is not None and self._is_3d_dataset():
            n_3d_participants = int(self._config.percentage_3d_participants * len(participant_ids))
            participant_ids = participant_ids[:n_3d_participants]
        self._participant_ids = participant_ids
        self._sample_metadatas = self._list_sample_metadatas()

        self._cached_data_adapters = None

        self._input_serial_rng = np.random.default_rng(config.seed)
        self._target_serial_rng = np.random.default_rng(config.seed + 1)
        self._target_timestep_rng = np.random.default_rng(config.seed + 2)
        self._input_timestep_rng = np.random.default_rng(config.seed + 3)

        if config.use_image_augmentations:
            self._image_transform = Compose([
                RandomApply(torch.nn.ModuleList([ColorJitter((0.6, 1.5), (0.6, 2.5), (0.5, 2.5), 0.1)]), p=0.8)])


    def _worker_init_fn(self, worker_id: int):
        # worker_info = torch.utils.data.get_worker_info()
        # dataset: BaseMVDataset = worker_info.dataset

        self._input_serial_rng = np.random.default_rng(self._config.seed * worker_id)
        self._target_serial_rng = np.random.default_rng(self._config.seed * worker_id + 1)
        self._target_timestep_rng = np.random.default_rng(self._config.seed * worker_id + 2)
        self._input_timestep_rng = np.random.default_rng(self._config.seed * worker_id + 3)

    @staticmethod
    def _concat_worker_init_fn(worker_id: int):
        worker_info = torch.utils.data.get_worker_info()
        dataset: ConcatDataset = worker_info.dataset
        for dset in dataset.datasets:
            dset._worker_init_fn(worker_id)

    @abstractmethod
    def _list_participant_ids(self) -> List[int]:
        pass

    @abstractmethod
    def _list_sample_metadatas(self) -> List[SampleMetadata]:
        pass

    @abstractmethod
    def _get_fixed_head_crop(self) -> Crop:
        pass

    @abstractmethod
    def _is_3d_dataset(self) -> bool:
        pass

    @abstractmethod
    def _get_data_adapter(self, sample_metadata: SampleMetadata, expression_code_config: ExpressionCodeConfig) -> BaseDataAdapter:
        pass

    def get_random_crop(self, image: np.ndarray) -> Crop:
        H, W, _ = image.shape
        min_w = min(max(self._config.target_resolution, self._config.min_input_crop_size), W)  # In NeRSemble, heads will realistically be at least 512x512
        min_h = self._config.target_resolution
        random_w = np.random.randint(min_w, W + 1)
        random_h = random_w
        random_x = np.random.randint(W - random_w + 1)
        random_y = np.random.randint(H - random_h + 1)

        crop = Crop(random_x, random_y, random_w, random_h)
        return crop

    def get_custom_crop(self, sample_metadata: SampleMetadata) -> Crop:
        raise NotImplementedError()

    def __getitem__(self, idx: int):
        idx_offset = 0
        for _ in range(100):
            try:
                return self._get_item(idx + idx_offset)
            except (FileNotFoundError, KeyError, IndexError) as e:
                print(e)
                idx_offset = (idx_offset + 1) % len(self)

        raise IllegalStateException(f"Retries failed too many times to load dataset sample {idx}: {self._sample_metadatas[idx]}")

    def _get_item(self, idx: int) -> MVDatasetSample:
        sample_metadata = self._sample_metadatas[idx]
        if self._config.use_cross_reenactment:
            sample_metadata_input = self._sample_metadatas[(idx + 1) % len(self._sample_metadatas)]
            inference_sample = self.get_inference_sample(sample_metadata_input)
        else:
            inference_sample = self.get_inference_sample(sample_metadata)

        return self.get_target_sample(sample_metadata, inference_sample)

    def get_target_sample(self, sample_metadata: SampleMetadata, inference_sample: MVDatasetInferenceSample) -> MVDatasetSample:
        data_adapter = self.get_data_adapter(sample_metadata)

        if self._config.target_view_sampling != 'same_as_input':
            available_cameras = data_adapter.list_cameras(sample_metadata)

        if self._config.target_timestep_sampling != 'same_as_input':
            available_timesteps = data_adapter.list_timesteps(sample_metadata)

        if self._config.target_timestep_sampling == 'same_as_input':
            target_timesteps = [sample_metadata.timestep for _ in range(self._config.n_target_timesteps)]
        elif self._config.target_timestep_sampling == 'random':
            target_timesteps = self._target_timestep_rng.choice(available_timesteps, self._config.n_target_timesteps, replace=False).tolist()
        elif self._config.target_timestep_sampling == 'evenly_spaced':
            target_timesteps = [available_timesteps[idx] for idx in
                                np.linspace(0, len(available_timesteps) - 1, self._config.n_target_timesteps, dtype=int)]
        elif self._config.target_timestep_sampling == 'middle':
            assert self._config.n_target_timesteps == 1
            target_timesteps = [available_timesteps[len(available_timesteps) // 2]]
        elif self._config.target_timestep_sampling == 'all':
            target_timesteps = available_timesteps

        input_serial = inference_sample.input_sample_metadatas[0].serial
        if self._config.target_view_sampling == 'random':

            if self._config.back_head_sample_weight is not None:
                back_head_serials = data_adapter.list_back_head_cameras()
                serial_weights = np.array([self._config.back_head_sample_weight if serial in back_head_serials else 1 for serial in available_cameras])
                serial_weights = serial_weights / serial_weights.sum()
                target_serials = self._target_serial_rng.choice(available_cameras, self._config.n_target_views, replace=False, p=serial_weights).tolist()
            else:
                target_serials = self._target_serial_rng.choice(available_cameras, self._config.n_target_views, replace=False).tolist()
        elif self._config.target_view_sampling == 'input_plus_random':
            if self._config.back_head_sample_weight is not None:
                back_head_serials = data_adapter.list_back_head_cameras()
                serial_weights = np.array([self._config.back_head_sample_weight if serial in back_head_serials else 1 for serial in available_cameras if serial != input_serial])
                serial_weights = serial_weights / serial_weights.sum()
                if self._config.n_target_views > 1:
                    target_serials = [input_serial] + self._target_serial_rng.choice([serial for serial in available_cameras if serial != input_serial], self._config.n_target_views - 1, replace=False, p=serial_weights).tolist()
                else:
                    target_serials = [input_serial]
            else:
                target_serials = [input_serial] + self._target_serial_rng.choice([serial for serial in available_cameras if serial != input_serial],
                                                                                 self._config.n_target_views - 1, replace=False).tolist()
        elif self._config.target_view_sampling == 'sequential':
            target_serials = available_cameras[:self._config.n_target_views]
        elif self._config.target_view_sampling == 'right':
            right_cameras = data_adapter.list_cameras_right(sample_metadata)
            right_cameras = [cam for cam in right_cameras if cam in available_cameras]
            target_serials = right_cameras[:self._config.n_target_views]
        elif self._config.target_view_sampling == 'eval':
            eval_cameras = data_adapter.list_cameras_eval(sample_metadata)
            target_serials = eval_cameras
        elif self._config.target_view_sampling == 'same_as_input':
            target_serials = [sample_metadata.serial]
        else:
            raise ValueError(f"Unknown target view sampling mode: {self._config.target_view_sampling}")

        render_cam2world_poses = []
        render_head_poses = []
        render_intrinsics = []
        target_images = []
        target_sample_metadatas = []
        target_masks = []
        expression_codes = []
        for serial in target_serials:
            if self._config.target_timestep_sampling == 'random_per_view':
                target_timesteps = self._target_timestep_rng.choice(available_timesteps, self._config.n_target_timesteps, replace=True).tolist()

            for timestep in target_timesteps:

                sample_metadata_target = replace(sample_metadata, serial=serial, timestep=timestep)
                target_image = data_adapter.load_image(sample_metadata_target)
                H, W, _ = target_image.shape
                mask = data_adapter.load_mask(sample_metadata_target)
                downscale_factor = self._config.target_resolution / W

                if self._config.use_random_target_cropping:
                    crop = self.get_random_crop(target_image)
                elif self._config.use_custom_target_cropping:
                    crop = self.get_custom_crop(sample_metadata_target)

                elif self._config.use_square_crops:
                    y_start = int(H / 2 - W / 2)
                    crop = Crop(0, y_start, W, W)
                else:
                    target_y = int(int((downscale_factor * H // 8) * 8) / downscale_factor)  # Ensure that height is a multiply of 8
                    crop = Crop(0, 0, W, target_y)

                if self._config.apply_color_correction:
                    target_image = crop.apply(target_image)
                    target_image = resize_img(target_image, self._config.target_resolution / crop.w)

                    mask = crop.apply(mask)
                    mask = resize_img(mask, self._config.target_resolution / crop.w)

                    target_image = data_adapter.apply_color_correction(target_image, serial)
                else:
                    target_image = crop.apply(target_image)
                    target_image = resize_img(target_image, downscale_factor)

                target_image = torch.tensor(target_image / 255., dtype=torch.float32).permute(2, 0, 1)
                target_images.append(target_image)

                cam2world_pose, intrinsics, head_pose = self.prepare_target_camera(sample_metadata_target, crop=crop)

                render_cam2world_poses.append(cam2world_pose)
                render_intrinsics.append(intrinsics)
                render_head_poses.append(head_pose)
                target_masks.append(torch.tensor(mask) / 255)
                target_sample_metadatas.append(SampleMetadata(sample_metadata.participant_id, sample_metadata.sequence_name, timestep, serial,
                                                              dataset=sample_metadata.dataset, environment=sample_metadata.environment))

                if self._config.load_expression_codes:
                    expression_code = data_adapter.load_expression_code(sample_metadata_target)
                    expression_code = torch.tensor(expression_code, dtype=torch.float32)
                    expression_codes.append(expression_code)

        if target_images:
            target_images = torch.stack(target_images)

        if target_masks:
            target_masks = torch.stack(target_masks)

        if self._config.normalize_images:
            # [0, 1] -> [-1, 1]
            target_images = target_images * 2 - 1

        render_bg_color = torch.tensor((255, 255, 255))
        if self._config.bg_color is not None:
            render_bg_color = torch.tensor(self._config.bg_color)
        elif self._config.use_random_bg_color:
            render_bg_color = (torch.rand(3) * 255).to(torch.uint8)

        if self._config.use_image_augmentations and inference_sample.input_sample_metadatas[0].dataset in self._config.augmentation_datasets:
            combined_images = torch.cat([inference_sample.input_images, target_images])
            combined_images = self._image_transform(combined_images)
            input_images, target_images = combined_images.split((inference_sample.input_images.shape[0], target_images.shape[0]))  # TODO: Apply masks again

            if self._config.mask_input_image:
                input_images = (inference_sample.input_masks[:, None] * input_images) + (1 - inference_sample.input_masks[:, None])  # Input images have always white background
        else:
            input_images = inference_sample.input_images

        # Apply target masks
        target_images = (target_masks[:, None] * target_images) + (1 - target_masks[:, None]) * (render_bg_color[None, :, None, None] / 255)

        dataset_ids = inference_sample.dataset_ids

        return MVDatasetSample(
            input_images=input_images,
            input_masks=None,
            input_sample_metadatas=inference_sample.input_sample_metadatas,
            input_cam2worlds=inference_sample.input_cam2worlds,
            input_intrinsics=inference_sample.input_intrinsics,
            input_view_mask=inference_sample.input_view_mask,
            input_expression_codes=inference_sample.input_expression_codes,
            render_cam2world_poses=render_cam2world_poses,
            render_intrinsics=render_intrinsics,
            render_resolution=Dimensions(target_images[0].shape[2], target_images[0].shape[1]) if len(target_images) > 0 else None,
            render_bg_color=tuple(render_bg_color),
            target_images=target_images,
            target_sample_metadatas=target_sample_metadatas,
            expression_codes=torch.stack(expression_codes) if expression_codes else None,
            render_head_poses=render_head_poses if self._config.load_render_head_poses else None,
            dataset_ids=dataset_ids,
        )

    def get_inference_sample(self, sample_metadata: SampleMetadata):
        data_adapter = self.get_data_adapter(sample_metadata)

        if sample_metadata.timestep != 'random':
            # Bring cameras into head-centric format
            model_to_world_no_scale, head_scale = data_adapter.load_head_pose(sample_metadata)

        n_input_views = self._config.n_input_views + self._config.n_input_views_condition

        if sample_metadata.serial == 'random':
            input_serials = self._input_serial_rng.choice(data_adapter.list_cameras(sample_metadata), n_input_views, replace=False)
        else:
            if n_input_views > 1:
                if isinstance(sample_metadata.serial, list):
                    input_serials = sample_metadata.serial
                else:
                    all_serials = data_adapter.list_cameras(sample_metadata)
                    serial_idx = all_serials.index(sample_metadata.serial)
                    input_serials = (all_serials * (int(ceil(n_input_views / len(all_serials))) + 1))[serial_idx: serial_idx + n_input_views]
            else:
                input_serials = [sample_metadata.serial]

        assert len(input_serials) == n_input_views, f"Got {input_serials} but expected {n_input_views} input views"

        input_images = []
        input_sample_metadatas = []
        input_cam2worlds = []
        input_intrinsics = []
        input_expression_codes = []
        input_masks = []

        if sample_metadata.timestep == 'random':
            if self._config.input_timestep_sampling == 'evenly_spaced':
                available_timesteps = data_adapter.list_timesteps(sample_metadata)
                input_timesteps = [available_timesteps[idx] for idx in np.linspace(0, len(available_timesteps) - 1, self._config.n_input_views, dtype=int)]
            else:
                available_timesteps = data_adapter.list_timesteps(sample_metadata)
                if len(available_timesteps) < n_input_views:
                    assert len(input_serials) > 1
                    input_timesteps = self._input_timestep_rng.choice(available_timesteps, n_input_views, replace=True).tolist()
                else:
                    input_timesteps = self._input_timestep_rng.choice(available_timesteps, n_input_views, replace=False).tolist()  # TODO: Abusing n_input_views here
        else:
            input_timesteps = [sample_metadata.timestep for _ in range(len(input_serials))]

        for input_serial, input_timestep in zip(input_serials, input_timesteps):

            sample_metadata_input = replace(sample_metadata, serial=input_serial, timestep=input_timestep)
            image_input = data_adapter.load_image(sample_metadata_input)
            input_cam2world_pose, input_intr = data_adapter.load_camera_params(sample_metadata_input)

            if sample_metadata.timestep == 'random':
                # Have to load separate head pose for each input view if different timesteps are being used
                model_to_world_no_scale, head_scale = data_adapter.load_head_pose(sample_metadata_input)

            if self._config.use_random_input_cropping:
                crop = self.get_random_crop(image_input)
            elif self._config.use_custom_input_cropping:
                crop = self.get_custom_crop(sample_metadata_input)
            else:
                crop = self._get_fixed_head_crop()

            image_cropped, input_intr = self.crop_and_resize_input_image(image_input, crop, input_intr, as_torch=False)

            if self._config.apply_color_correction:
                image_cropped = data_adapter.apply_color_correction(image_cropped, input_serial)

            if self._config.mask_input_image:
                mask = data_adapter.load_mask(sample_metadata_input)
                mask = self.crop_and_resize_input_image(mask, crop, None, as_torch=False)
                image_cropped = self._apply_mask(image_cropped, mask)

                input_masks.append(torch.tensor(mask) / 255)

            image_cropped = torch.tensor(image_cropped / 255., dtype=torch.float32).permute(2, 0, 1)
            input_intr = input_intr.rescale(1 / self._config.input_resolution, inplace=False)

            # Important order: First rotate, then scale
            input_cam2world_pose = Pose(
                model_to_world_no_scale.invert().numpy() @ input_cam2world_pose,
                pose_type=PoseType.CAM_2_WORLD)
            input_cam2world_pose.set_translation(head_scale * input_cam2world_pose.get_translation())

            if self._config.normalize_images:
                # [0, 1] -> [-1, 1]
                image_cropped = image_cropped * 2 - 1

            input_images.append(image_cropped)
            input_sample_metadatas.append(sample_metadata_input)
            input_cam2worlds.append(input_cam2world_pose)
            input_intrinsics.append(input_intr)

            if self._config.load_input_expression_codes:
                expression_code = data_adapter.load_expression_code(sample_metadata_input)
                expression_code = torch.tensor(expression_code, dtype=torch.float32)
                input_expression_codes.append(expression_code)

        input_view_mask = torch.ones((n_input_views,), dtype=bool)

        dataset_ids = [1 if input_sample_metadatas[0].dataset in ['ava256', 'nersemble', 'cafca'] else 0]

        return MVDatasetInferenceSample(
            input_images=torch.stack(input_images),
            input_sample_metadatas=input_sample_metadatas,
            input_cam2worlds=input_cam2worlds,
            input_intrinsics=input_intrinsics,
            input_view_mask=input_view_mask,
            input_masks=torch.stack(input_masks) if input_masks else None,
            input_expression_codes=torch.stack(input_expression_codes) if input_expression_codes else None,
            dataset_ids=torch.tensor(dataset_ids),
        )

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray):
        # Apply background removal separately as otherwise color correction changes colors of background as well
        mask = np.expand_dims(mask, axis=2)

        image = image / 255.
        mask = mask / 255.
        image = mask * image + (1 - mask) * np.ones_like(image) * np.array([1.0, 1.0, 1.0])[None, None]

        image = image * 255.
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

        return image

    def crop_and_resize_input_image(self, input_image: np.ndarray, crop: Crop, input_intrinsics: Optional[Intrinsics] = None, as_torch: bool = True):
        image_cropped = crop.apply(input_image)
        image_cropped = resize_img(image_cropped, self._config.input_resolution / crop.w)
        if as_torch:
            image_cropped = torch.tensor(image_cropped / 255., dtype=torch.float32).permute(2, 0, 1)

        if input_intrinsics is not None:
            input_intrinsics.crop(crop_left=crop.x, crop_top=crop.y)
            input_intrinsics.rescale(self._config.input_resolution / crop.w)
            return image_cropped, input_intrinsics
        else:
            return image_cropped

    def get_data_adapter(self, sample_metadata: SampleMetadata) -> BaseDataAdapter:
        if self._cached_data_adapters is None:
            self._cached_data_adapters = defaultdict(lambda: defaultdict(dict))

        if sample_metadata.sequence_name not in self._cached_data_adapters[sample_metadata.dataset][sample_metadata.participant_id]:
            expression_code_config = self._config.expression_code_config
            data_adapter = self._get_data_adapter(sample_metadata, expression_code_config)
            self._cached_data_adapters[sample_metadata.dataset][sample_metadata.participant_id][sample_metadata.sequence_name] = data_adapter
        else:
            data_adapter = self._cached_data_adapters[sample_metadata.dataset][sample_metadata.participant_id][sample_metadata.sequence_name]

        return data_adapter

    def prepare_target_camera(self, sample_metadata: SampleMetadata, crop: Optional[Crop] = None) -> Tuple[Pose, Intrinsics, Pose]:
        data_adapter = self.get_data_adapter(sample_metadata)

        # Bring cameras into head-centric format
        model_to_world_no_scale, head_scale = data_adapter.load_head_pose(sample_metadata)

        cam2world_pose, intrinsics = data_adapter.load_camera_params(sample_metadata)
        # Important order: First rotate, then scale
        cam2world_pose = Pose(
            model_to_world_no_scale.invert().numpy() @ cam2world_pose,
            pose_type=PoseType.CAM_2_WORLD)
        cam2world_pose.set_translation(head_scale * cam2world_pose.get_translation())

        if self._config.use_random_target_cropping or self._config.use_square_crops or crop is not None:
            intrinsics = intrinsics.crop(crop.x, crop.y, inplace=False)
            intrinsics = intrinsics.rescale(self._config.target_resolution / crop.w, inplace=False)

        return cam2world_pose, intrinsics, model_to_world_no_scale

    def clear_cache(self):
        self._cached_data_adapters = None

    def __len__(self) -> int:
        return len(self._sample_metadatas)

    @staticmethod
    def collate_fn_inference(samples: List[MVDatasetInferenceSample]) -> FlexAvatarBatch:
        batched_values = dict()
        for field in fields(samples[0]):
            values = []
            for sample in samples:
                value = getattr(sample, field.name)
                if value is not None:
                    values.append(value)

            if len(values) == 0:
                values = None
            elif isinstance(values[0], torch.Tensor):
                values = torch.stack(values)

            batched_values[field.name] = values

        batch = FlexAvatarBatch(**batched_values)
        return batch

    @staticmethod
    def collate_fn(samples: List[MVDatasetSample]) -> FlexAvatarBatch:
        batched_values = dict()
        for field in fields(samples[0]):
            values = []
            for sample in samples:
                value = getattr(sample, field.name)
                if value is not None:
                    values.append(value)

            if len(values) == 0:
                values = None
            elif isinstance(values[0], torch.Tensor):
                values = torch.stack(values)

            batched_values[field.name] = values

        batch = FlexAvatarBatch(**batched_values)
        return batch
