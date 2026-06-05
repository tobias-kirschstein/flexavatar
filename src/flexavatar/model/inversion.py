import platform
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable

import numpy as np
import torch
from dreifus.graphics import Dimensions
from dreifus.image import torch_to_numpy_img
from elias.config import better_replace
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from sam_loss import SAMLoss
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from flexavatar.config.dataset_config import FlexAvatarBatch
from flexavatar.model.flexavatar_model import FlexAvatarModel, FlexAvatarOutput
from flexavatar.util.dino_loss import DinoV2Loss


@dataclass
class FittingConfig:
    steps: int = 200
    lr: float = 1e-2
    lr_min: float = 5e-4
    lambda_sam_loss: float = 1
    lambda_dino_loss: float = 1
    lambda_latent_reg: float = 0
    from_scratch: bool = False
    optimize_expression_code: bool = True


class FittingManager:

    def __init__(self, model: FlexAvatarModel, inversion_config: FittingConfig = FittingConfig()):
        self._model = model

        if inversion_config.lambda_sam_loss > 0:
            self._sam_criterion = SAMLoss()

            if model._config.compile and platform.system() == 'Linux':
                self._sam_criterion.compile()

        if inversion_config.lambda_dino_loss > 0:
            self._dino_criterion = DinoV2Loss()

            if model._config.compile and platform.system() == 'Linux':
                self._dino_criterion.compile()

        self._config = inversion_config

    def run_inversion(self,
                      render_batch: FlexAvatarBatch,
                      inversion_callback: Callable[[FlexAvatarOutput], None] = lambda output: None) -> Tuple[torch.Tensor, List[np.ndarray], List[float]]:
        render_batches_inversion = []
        n_inputs = self._model._config.n_input_views
        for i in range(0, render_batch.VI, n_inputs):
            render_batch_inversion = better_replace(render_batch,
                                                    input_images=render_batch.input_images[:, i:i+ n_inputs],
                                                    input_expression_codes=render_batch.input_expression_codes[:, i:i+ n_inputs],
                                                    features=render_batch.features[:, i:i + n_inputs],
                                                    input_cam2worlds=[render_batch.input_cam2worlds[0][i:i+ n_inputs]],
                                                    input_intrinsics=[render_batch.input_intrinsics[0][i:i+ n_inputs]],
                                                    render_cam2world_poses=[render_batch.input_cam2worlds[0][i: i + n_inputs]],
                                                    render_intrinsics=[[intr.rescale(render_batch.input_images.shape[-2], inplace=False)
                                                                        for intr in intr_list[i: i + n_inputs]]
                                                                       for intr_list
                                                                       in render_batch.input_intrinsics],
                                                    render_resolution=[Dimensions(input_images.shape[-1], input_images.shape[-2])
                                                                       for input_images in render_batch.input_images],
                                                    expression_codes=render_batch.input_expression_codes[:, i: i + n_inputs],
                                                    render_bg_color=[(255, 255, 255)])
            render_batches_inversion.append(render_batch_inversion)

        output_internal = self._model.forward(render_batches_inversion[0], only_internal_representations=True)
        with torch.enable_grad():
            latent_avatar_code_init = output_internal.gaussian_models_output.internal_representations.clone()

            if self._config.from_scratch:
                latent_avatar_code_init = torch.zeros_like(latent_avatar_code_init)

            latent_avatar_code = nn.Parameter(latent_avatar_code_init)
            params = [latent_avatar_code]

            if self._config.optimize_expression_code:
                expression_code_offsets = []
                for render_batch_inversion in render_batches_inversion:
                    expression_code_offset = nn.Parameter(torch.zeros_like(render_batch_inversion.expression_codes))
                    render_batch_inversion.expression_codes = render_batch_inversion.expression_codes + expression_code_offset
                    expression_code_offsets.append(expression_code_offset)
                params.extend(expression_code_offsets)

            optimizer = Adam(params, lr=self._config.lr)
            scheduler = CosineAnnealingLR(optimizer, self._config.steps, eta_min=self._config.lr_min)
            for p in self._model.parameters():
                p.requires_grad = False


            progress = tqdm(range(self._config.steps), desc="Fitting Avatar Code")
            losses = []
            inversion_images = []
            for j in progress:
                i_input = np.random.randint(len(render_batches_inversion))
                render_batch_inversion = render_batches_inversion[i_input]
                target_images = render_batch_inversion.input_images

                output_inversion = self._model.forward(render_batch_inversion, cached_internal_representations=latent_avatar_code)
                rendered_images = output_inversion.rendering_output.rendered_images
                loss = 0.8 * l1_loss(rendered_images, target_images) + 0.2 * (
                        1 - ssim(rendered_images.flatten(0, 1), target_images.flatten(0, 1)))

                if self._config.lambda_sam_loss > 0:
                    sam_loss = self._sam_criterion(rendered_images.flatten(0, 1), target_images.flatten(0, 1))
                    loss = loss + self._config.lambda_sam_loss * sam_loss

                if self._config.lambda_dino_loss > 0:
                    dino_loss = self._dino_criterion(rendered_images.flatten(0, 1), target_images.flatten(0, 1))
                    loss = loss + self._config.lambda_dino_loss * dino_loss

                if self._config.lambda_latent_reg > 0:
                    latent_reg = (latent_avatar_code - latent_avatar_code_init).square().mean()
                    loss = loss + self._config.lambda_latent_reg * latent_reg

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                progress.set_postfix({'loss': loss.item()})
                inversion_images.append(torch_to_numpy_img(rendered_images[0, 0]))
                losses.append(loss.item())

                inversion_callback(output_inversion)

        return latent_avatar_code, inversion_images, losses
