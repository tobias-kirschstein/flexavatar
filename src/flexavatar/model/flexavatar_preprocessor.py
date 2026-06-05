import platform
from typing import List, Callable

import torch
import torchvision

from flexavatar.config.dataset_config import MVDatasetConfig, FlexAvatarBatch, SampleMetadata
from flexavatar.model.dinov2 import DinoV2
from flexavatar.util.lru_cache import DeviceLRUCache


class FlexAvatarPreprocessor:

    def __init__(self,
                 dataset_config: MVDatasetConfig,
                 use_caching: bool = False,
                 cache_dtype: torch.dtype = torch.float32,
                 cache_size: int = 50_000,
                 compile: bool = False,
                 use_bfloat16: bool = False):
        self._dataset_config = dataset_config
        self._use_caching = use_caching
        self._cache_dtype = cache_dtype
        self._use_bfloat16 = use_bfloat16
        device = torch.device('cuda')

        if dataset_config.use_dino and not dataset_config.load_precomputed_dino:
            self._dino = DinoV2(dataset_config.dino_name)

            for p in self._dino.parameters():
                p.requires_grad = False

            if compile and platform.system() == 'Linux':
                self._dino.compile()

            if use_caching:
                self._dino_cache = DeviceLRUCache(device, torch.device('cpu'), cache_dtype=cache_dtype, max_size=cache_size)


    def process(self, batch: FlexAvatarBatch, disable_cache: bool = False, enable_grads: bool = False):
        use_caching = self._use_caching and not disable_cache
        device = batch.input_images.device
        if self._dataset_config.use_dino and not self._dataset_config.load_precomputed_dino:
            batched_feature_images = []

            if use_caching:
                cached_idxs = []
                non_cached_idxs = []
                non_cached_images = []
                non_cached_sample_metadatas = []

                for b, input_sample_metadata in enumerate(batch.input_sample_metadatas):
                    feature_input_images = []
                    for v in range(len(input_sample_metadata)):
                        if input_sample_metadata[v] in self._dino_cache:
                            cached_idxs.append((b, v))
                            feature_image = self._dino_cache.get(input_sample_metadata[v])
                            feature_input_images.append(feature_image)
                        else:
                            non_cached_idxs.append((b, v))
                            non_cached_images.append(batch.input_images[b][v])
                            non_cached_sample_metadatas.append(input_sample_metadata[v])
                            feature_input_images.append(None)

                    batched_feature_images.append(feature_input_images)

                if non_cached_images:
                    non_cached_images = torch.stack(non_cached_images)
                    non_cached_features = self._get_dino_features(non_cached_images)

                    for (b, v), non_cached_sample_metadata, non_cached_feature in zip(non_cached_idxs, non_cached_sample_metadatas, non_cached_features):
                        self._dino_cache.put(non_cached_sample_metadata, non_cached_feature)
                        batched_feature_images[b][v] = non_cached_feature

                batched_feature_images = torch.stack([torch.stack(feature_images) for feature_images in batched_feature_images])

            else:
                for input_images in batch.input_images:
                    feature_images = self._get_dino_features(input_images, enable_grads=enable_grads)
                    batched_feature_images.append(feature_images)

                batched_feature_images = torch.stack(batched_feature_images)
            batch.features = batched_feature_images

        return batch

    def _get_dino_features(self, input_image: torch.Tensor, enable_grads: bool = False) -> torch.Tensor:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self._use_bfloat16):
            input_image = torchvision.transforms.Resize((self._dataset_config.dino_resolution, self._dataset_config.dino_resolution))(input_image)
            feature_image = self._dino(input_image, enable_grads=enable_grads)
        return feature_image

    def _perform_cached_lookup(self,
                               cache: DeviceLRUCache,
                               transform_fn: Callable[[torch.Tensor], torch.Tensor],
                               images: torch.Tensor,
                               sample_metadatas: List[List[SampleMetadata]]):
        cached_idxs = []
        non_cached_idxs = []
        non_cached_images = []
        non_cached_sample_metadatas = []

        batched_transformed_images = []

        for b, sample_metadata in enumerate(sample_metadatas):
            transformed_images = []
            for v in range(len(sample_metadata)):
                if sample_metadata[v] in cache:
                    cached_idxs.append((b, v))
                    feature_image = cache.get(sample_metadata[v])
                    transformed_images.append(feature_image)
                else:
                    non_cached_idxs.append((b, v))
                    non_cached_images.append(images[b][v])
                    non_cached_sample_metadatas.append(sample_metadata[v])
                    transformed_images.append(None)

            batched_transformed_images.append(transformed_images)

        if non_cached_images:
            non_cached_images = torch.stack(non_cached_images)
            non_cached_transformed_images = transform_fn(non_cached_images)

            for (b, v), non_cached_sample_metadata, non_cached_transformed_image in zip(non_cached_idxs, non_cached_sample_metadatas,
                                                                                        non_cached_transformed_images):
                cache.put(non_cached_sample_metadata, non_cached_transformed_image)
                batched_transformed_images[b][v] = non_cached_transformed_image

        batched_transformed_images = torch.stack(
            [torch.stack(transformed_images) if transformed_images else torch.empty((0,)) for transformed_images in batched_transformed_images])

        return batched_transformed_images
