from dataclasses import dataclass, field
from math import ceil, floor
from typing import List, Literal

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn

DinoDistanceType = Literal['l1', 'l2', 'cos']
DinoProvider = Literal['diffusers', 'torchhub']

@dataclass
class DinoV2LossConfig:
    dino_model: str = 'facebook/dinov2-small'  # dinov2_vits14
    dino_layers: List[int] = field(default_factory=lambda: [5, 8, 11])
    dino_distance: DinoDistanceType = 'l1'
    dino_img_size: int = 512
    dino_provider: DinoProvider = 'diffusers'

class DinoV2Loss(nn.Module):

    def __init__(self, config: DinoV2LossConfig = DinoV2LossConfig()):
        super().__init__()
        if config.dino_provider == 'diffusers':
            from diffusers import AutoModel
            self._dino_model = AutoModel.from_pretrained(config.dino_model)
        elif config.dino_provider == 'torchhub':
            self._dino_model = torch.hub.load('facebookresearch/dinov2', config.dino_model)
        else:
            raise ValueError(f"Unknown dino loss provider: {config.dino_provider}")

        self._dino_model = self._dino_model.cuda()
        self._dino_model.eval()
        for p in self._dino_model.parameters():
            p.requires_grad = False
        self._config = config

        crop_length = config.dino_img_size - (config.dino_img_size // 14) * 14
        crop_length_1 = int(ceil(crop_length / 2))
        crop_length_2 = int(floor(crop_length / 2))


        self._image_transforms = transforms.Compose([
            transforms.Resize(config.dino_img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Pad([-crop_length_1, -crop_length_1, -crop_length_2, -crop_length_2]),  # Center crop, s.t. image is multiple of 14
            transforms.CenterCrop((config.dino_img_size // 14) * 14),  # Center crop, s.t. image is multiple of 14
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def forward(self, predicted_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
            predicted_images: [B, 3, H, W] in [0, 1] range
            target_images: [B, 3, H, W] in [0, 1] range

        Returns
        -------
            scalar with Dino loss
        """

        rendered_features = self._get_features(predicted_images)

        with torch.no_grad():
            target_features = self._get_features(target_images)

        if self._config.dino_distance == "l1":
            dino_losses = F.l1_loss(
                rendered_features,
                target_features.detach(),
                reduction="mean"
            )
        elif self._config.dino_distance == "l2":
            dino_losses = F.mse_loss(
                rendered_features,
                target_features.detach(),
                reduction="mean"
            )
        elif self._config.dino_distance == 'cos':
            rendered_features = F.normalize(rendered_features, dim=-1)
            target_features = F.normalize(target_features.detach(), dim=-1)
            dino_losses = 1 - (rendered_features * target_features).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unknown dino loss distance: {self._config.dino_distance}")

        return dino_losses

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        # H_dino = (x.shape[2] // 14) * 14
        # W_dino = (x.shape[3] // 14) * 14
        # x = x[:, :, :H_dino, :W_dino]  # throw out some pixels at the bottom and right to make the image a multiple of 14

        if self._config.dino_provider == 'diffusers':
            features = self._dino_model(pixel_values=self._image_transforms(x), output_hidden_states=True).hidden_states
            features = torch.cat([features[i] for i in self._config.dino_layers], dim=1)
        elif self._config.dino_provider == 'torchhub':
            features = self._dino_model.get_intermediate_layers(self._image_transforms(x), n=self._config.dino_layers, reshape=True)
            features = torch.cat(features, dim=1)
        else:
            raise ValueError(f"Unknown dino loss provider: {self._config.dino_provider}")

        return features