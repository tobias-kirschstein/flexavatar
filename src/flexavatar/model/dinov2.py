from math import sqrt

import torch
from torch import nn
from torchvision.transforms import Normalize


class DinoV2(nn.Module):

    def __init__(self, dino_name: str = 'dinov2_vitg14_reg'):
        super().__init__()
        self._dino = torch.hub.load('facebookresearch/dinov2:85a24602099d397264d5b30461ad7f3bfd726ca1', dino_name,
                                    trust_repo=True, skip_validation=True)
        self._dino.eval()
        self._dino.cuda()

        self._normalizer = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, images: torch.Tensor, enable_grads: bool = False) -> torch.Tensor:
        B = images.shape[0]
        with torch.set_grad_enabled(enable_grads):
            images = self._normalizer(images)
            output = self._dino.forward_features(images)
            patch_tokens = output['x_norm_patchtokens']
            res = int(sqrt(patch_tokens.shape[1]))
            feature_image = patch_tokens.reshape(B, res, res, patch_tokens.shape[-1])
            feature_image = feature_image.permute(0, 3, 1, 2)

        return feature_image

    def get_embed_dim(self) -> int:
        return self._dino.embed_dim

def get_dino_embed_dim(dino_name: str):
    if dino_name == 'dinov2_vitg14_reg':
        return 1536
    elif dino_name == 'dinov2_vits14_reg' or dino_name == 'dinov3-vits16-pretrain-lvd1689m':
        return 384
    elif dino_name == 'dinov2_vitb14_reg':
        return 768
    else:
        raise ValueError(f"Unknown dino name: {dino_name}")
        return 1