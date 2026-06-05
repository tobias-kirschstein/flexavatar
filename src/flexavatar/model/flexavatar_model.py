import platform
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict

import numpy as np
import torch
from dreifus.matrix import Pose, Intrinsics
from einops import rearrange
from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render_distwar, render_gsplat_batched
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import pose_to_rendercam
from torch import nn
from torch.nn import GELU, LayerNorm, PixelShuffle, Identity
from torch.nn.modules.module import T
from torchvision.ops import MLP
from trimesh import load_mesh

from flexavatar.config.dataset_config import FlexAvatarBatch
from flexavatar.config.flexavatar_config import HeadTransformerConfig, HeadTransformerType, CrossAttentionType, GaussianDecoderConfig, FlexAvatarModelConfig
from flexavatar.env import ASSETS_PATH
from flexavatar.model.lam_gs_layer import GSLayer
from flexavatar.model.lam_point_embedder import PointEmbed
from flexavatar.model.lam_transformer import TransformerDecoder
from flexavatar.model.nanogpt import GPTConfig, GPT
from flexavatar.model.stylegan_upsampler import StyleGANUpsamplerConfig, StyleGANPixelShuffleUpsampler
from flexavatar.util.plucker import plucker_embedder
from flexavatar.util.uv import gen_tritex


@dataclass
class RenderingOutput:
    rendered_images: torch.Tensor  # [B, TV, C, H, W]
    rendered_raw_images: Optional[torch.Tensor] = None


@dataclass
class GaussianModelsOutput:
    gaussian_models: List[List[GaussianModel]]
    gaussian_predictions: Optional[Dict[str, torch.Tensor]] = None
    internal_representations: Optional[torch.Tensor] = None


@dataclass
class FlexAvatarOutput:
    gaussian_models_output: GaussianModelsOutput
    rendering_output: RenderingOutput


def sample_template_positions(resolution: int, template_name: str = 'gghead_template') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    template_mesh = load_mesh(f"{ASSETS_PATH}/{template_name}.obj")
    if hasattr(template_mesh.visual, 'uv'):  # Assumes
        vt = template_mesh.visual.uv
        ft = template_mesh.faces
    else:
        # Trimesh cannot load / represent proper texel coordinates (vt / ft). Hence, they have to be stored separately and loaded here
        vtft = np.load(f"{ASSETS_PATH}/{template_name}_vtft.npz")
        vt = vtft['vt']
        ft = vtft['ft']
    uv_coords = vt
    faces = template_mesh.faces
    idxim, tidxim, barim = gen_tritex(uv_coords, faces, ft, resolution)
    vertices = template_mesh.vertices

    v0_map = vertices[idxim[..., 0]]
    v1_map = vertices[idxim[..., 1]]
    v2_map = vertices[idxim[..., 2]]
    flame_position_map = barim[..., [0]] * v0_map + barim[..., [1]] * v1_map + barim[..., [2]] * v2_map  # Maps texels to 3D positions

    xs = torch.linspace(-1, 1, steps=resolution)
    ys = torch.linspace(-1, 1, steps=resolution)

    xs, ys = torch.meshgrid(xs, ys, indexing='ij')
    sampled_uv_coords = torch.stack([ys, xs], dim=-1)

    torch_position_map = torch.from_numpy(flame_position_map).float().permute(2, 0, 1)  # [3, H_map, W_map]
    torch_face_index_map = torch.from_numpy(idxim).permute(2, 0, 1)
    valid_uv_map = (torch_face_index_map > 0).any(dim=0).float()[None]  # [1, H_map, W_map]

    valid_samples = torch.nn.functional.grid_sample(valid_uv_map.unsqueeze(0), sampled_uv_coords.unsqueeze(0))[0].permute(1, 2, 0)
    valid_samples = valid_samples[:, :, 0] > 0.99
    valid_uv_coords = sampled_uv_coords[valid_samples]  # [G, 2]
    uv_samples = valid_uv_coords.unsqueeze(0).unsqueeze(2)  # [1, G, 1, 2]
    sampled_positions = torch.nn.functional.grid_sample(torch_position_map.unsqueeze(0), uv_samples)[0, :, :, 0].T  # [G, 3]
    uv_samples = uv_samples[0, :, 0]

    return sampled_positions, uv_samples, torch_position_map.permute(1, 2, 0)


class HeadTransformer(nn.Module):

    def __init__(self, config: HeadTransformerConfig):
        super().__init__()
        self._config = config

        initial_gaussian_positions, _, position_map = sample_template_positions(config.res_head_tokens, config.head_template)

        if config.head_transformer_type == HeadTransformerType.MESH_TOKENS:
            self._head_token_embeddings = nn.Parameter(torch.zeros((len(initial_gaussian_positions), 1, config.transformer.d_hidden)))  # [HT, 1, D]
            initial_head_xyz = initial_gaussian_positions[None]
        elif config.head_transformer_type == HeadTransformerType.UV_TEXTURE:
            self._head_token_embeddings = nn.Parameter(torch.zeros((config.res_head_tokens ** 2, 1, config.transformer.d_hidden)))  # [HT, 1, D]
            initial_head_xyz = position_map.reshape(1, config.res_head_tokens ** 2, 3)
        else:
            raise ValueError(f"Unknown head transformer type: {config.head_transformer_type}")

        if config.use_lam_point_embedder:
            self.register_buffer("_initial_head_xyz", initial_head_xyz, persistent=False)
            if config.use_lam_point_embedder:
                self._query_point_embedder = PointEmbed(dim=config.transformer.d_hidden)

        if config.use_lam_transformer:
            self._transformer = TransformerDecoder('sd3_cond', config.transformer.n_layers, config.transformer.n_heads, config.transformer.d_hidden,
                                                   cond_dim=config.transformer.d_hidden,
                                                   use_ada_ln=False,
                                                   transform_keys=False)

        if config.d_expression_codes is not None:
            self._expression_mlp = MLP(config.d_expression_codes,
                                       [256] * 2 + [
                                           config.transformer.d_hidden if config.n_expression_tokens is None else config.transformer.d_hidden * config.n_expression_tokens],
                                       activation_layer=torch.nn.ReLU)

        if config.d_expression_codes is not None:
            if config.use_lam_transformer:
                self._expression_transformer = TransformerDecoder('sd3_cond',
                                                                  config.n_layers_expression_transformer, config.transformer.n_heads,
                                                                  config.transformer.d_hidden,
                                                                  cond_dim=config.transformer.d_hidden,
                                                                  use_ada_ln=False,
                                                                  transform_keys=False)

        if self._config.use_dataset_ids:
            n_dataset_ids = 2
            self._dataset_embedding = nn.Embedding(n_dataset_ids, config.transformer.d_hidden)

        nn.init.trunc_normal_(self._head_token_embeddings)

    def forward(self, x: torch.Tensor,
                expression_codes: Optional[torch.Tensor] = None,
                dataset_ids: Optional[torch.Tensor] = None,
                cached_internal_representations: Optional[torch.Tensor] = None,
                only_internal_representations: bool = False,
                ) -> Union[torch.Tensor, torch.Tensor]:

        if cached_internal_representations is None:
            B = x.shape[1]
            if not self._config.use_head_tokens:
                queries = torch.empty((0, B, self._config.transformer.d_hidden), device=x.device, dtype=x.dtype)
            elif self._config.use_lam_point_embedder:
                queries = self._query_point_embedder(self._initial_head_xyz).permute(1, 0, 2).repeat(1, B, 1).to(x.dtype)
            else:
                queries = self._head_token_embeddings.repeat(1, B, 1).to(x.dtype)

            if self._config.cross_attention_type == CrossAttentionType.Q2K:
                x = self._transformer(queries, keys=x)
            elif self._config.cross_attention_type == CrossAttentionType.Q2QK:
                qk = torch.cat([queries, x], dim=0)
                x = self._transformer(queries, keys=qk)
            elif self._config.cross_attention_type == CrossAttentionType.QK2QK:
                qk = torch.cat([queries, x], dim=0)
                x = self._transformer(qk)
                x = x[:len(queries)]
            else:
                raise ValueError(f"Unknown cross attention type: {self._config.cross_attention_type}")
        else:
            x = cached_internal_representations[:self._head_token_embeddings.shape[0]]

        internal_representation = x

        if only_internal_representations:
            return x, internal_representation

        B = x.shape[1]

        if self._config.d_expression_codes is not None:
            if self._config.n_expression_tokens is None:
                expression_tokens = self._expression_mlp(expression_codes).flatten(0, 1)
            else:
                expression_tokens = self._expression_mlp(expression_codes).reshape(B * expression_codes.shape[1],
                                                                                   self._config.n_expression_tokens,
                                                                                   self._config.transformer.d_hidden)
            expression_tokens = expression_tokens.permute(1, 0, 2)

            # Duplicate internal 3D representation for each expression code -> there will be separate GaussianModels for each expression code
            x = x.repeat_interleave(expression_codes.shape[1], dim=1)

            if self._config.use_dataset_ids:
                dataset_tokens = self._dataset_embedding(dataset_ids)
                dataset_tokens = dataset_tokens.permute(1, 0, 2)  # [1, B, D]
                dataset_tokens = dataset_tokens.repeat_interleave(expression_codes.shape[1], dim=1)
                expression_tokens = torch.cat([dataset_tokens, expression_tokens], dim=0)

        if self._config.d_expression_codes:
            x = self._expression_transformer(x, keys=expression_tokens)

        return x, internal_representation


class GaussianDecoder(nn.Module):
    def __init__(self, config: GaussianDecoderConfig):
        super().__init__()
        self._config = config

        self._n_color_channels = config.n_channels_color

        d_feature_maps = config.d_hidden
        mlp_d_in = d_feature_maps
        if config.head_transformer_type == HeadTransformerType.UV_TEXTURE and config.upscale_uv_texture is not None:
            mlp_d_in = mlp_d_in // config.upscale_uv_texture ** 2
            assert mlp_d_in * config.upscale_uv_texture ** 2 == d_feature_maps, "MLP input size needs to be divisible by upscale factor"

        if config.use_norm_before_mlp:
            self._layer_norm = LayerNorm(mlp_d_in)

        self._mlp_decoder = self.create_mlp_decoder(config, mlp_d_in)

        if config.head_transformer_type == HeadTransformerType.MESH_TOKENS:
            initial_gaussian_positions, uv_samples, _ = sample_template_positions(config.res_head_tokens, config.head_template)
        elif config.head_transformer_type == HeadTransformerType.UV_TEXTURE:
            initial_gaussian_positions, uv_samples, _ = sample_template_positions(
                config.res_head_tokens * config.upscale_uv_texture * config.oversampling_factor, config.head_template)

            if config.upscale_uv_texture is not None:
                stylegan_config = StyleGANUpsamplerConfig(
                    input_res=config.res_head_tokens,
                    output_res=config.upscale_uv_texture * config.res_head_tokens,
                    input_channels=d_feature_maps,
                    output_channels=d_feature_maps,
                    use_noise=False,
                    initialize_with_image=config.initialize_with_image
                )
                if config.use_stylegan_pixelshuffle_upsampler:
                    self._stylegan_upsampler = StyleGANPixelShuffleUpsampler(stylegan_config)
                else:
                    self._uv_texture_pixel_shuffle = PixelShuffle(config.upscale_uv_texture)

        else:
            raise ValueError(f"Unknown head transformer type: {config.head_transformer_type}")

        initial_gaussian_positions = initial_gaussian_positions[None]  # [1, G, 3]
        uv_samples = uv_samples[None, :, None]  # [1, G, 1, 2]
        self.register_buffer("_initial_gaussian_positions", initial_gaussian_positions, persistent=False)
        self.register_buffer("_uv_samples", uv_samples, persistent=False)

        self.register_buffer("_device_indicator", torch.empty(0), persistent=False)

    @property
    def device(self):
        return self._device_indicator.device

    def create_mlp_decoder(self, config: GaussianDecoderConfig, mlp_d_in: int, n_position_channels: int = 3):

        if config.use_lam_gs_decoder:
            mlp_decoder = GSLayer(mlp_d_in, sh_degree=self._config.sh_degree, use_rgb=self._config.sh_degree == 0)
        else:
            mlp_decoder = MLP(mlp_d_in,
                              [config.d_hidden] * (config.n_mlp_layers - 1) + [
                                  (n_position_channels + 3 + 4 + 1 + self._n_color_channels) * config.n_gaussians_per_token],
                              activation_layer=GELU)
            for p in mlp_decoder.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0, std=0.02)

        return mlp_decoder

    def _decode_gaussians(self,
                          mlp_decoder: nn.Module,
                          x: torch.Tensor,
                          h: Optional[int] = None,
                          v: int = 1,
                          perform_sampling: bool = True):
        B, HT, C = x.shape

        if self._config.head_transformer_type == HeadTransformerType.MESH_TOKENS:
            x = self._layer_norm(x)
            x = mlp_decoder(x)  # [B, HT, G * D_G]
            # x = self._mlp_decoder(self._dummy_tokens)
            x = rearrange(x, 'b t (g d) -> b (t g) d', g=self._config.n_gaussians_per_token)

            # x = self._dummy_mlp(self._dummy_mlp_input)
            # x = rearrange(x, 'b (g d) -> b g d', g=64000)


        elif self._config.head_transformer_type == HeadTransformerType.UV_TEXTURE:
            # [B*E, V*H*W, D]
            x = rearrange(x, 'b (v h w) d -> (b v) h w d', h=self._config.res_head_tokens if h is None else h, v=v)  # [B*E*V, H*W, D]

            uv_texture = x.permute(0, 3, 1, 2)  # [BV, C, H, W]

            sampled_features = self._upsample_feature_map(uv_texture, perform_sampling=perform_sampling)

            sampled_features = sampled_features.reshape(B, v * sampled_features.shape[-2], sampled_features.shape[-1])  # [B*E, V*H*W, D]

            if self._config.use_norm_before_mlp:
                x = self._layer_norm(sampled_features)
            else:
                x = sampled_features
            x = mlp_decoder(x)  # [B, G, D_G]
            if not self._config.use_lam_gs_decoder:
                x = rearrange(x, '(b v) t (g d) -> b (v t g) d', v=v, g=self._config.n_gaussians_per_token)

        else:
            raise ValueError(f"Unknown head transformer type: {self._config.head_transformer_type}")

        colors_sh = None
        if self._config.use_lam_gs_decoder:
            positions = x['xyz']
            scales = x['scaling']
            rotations = x['rotation']
            opacities = x['opacity']
            colors = x['shs'][:, :, 0]
            colors_sh = x['shs'][:, :, 1:]
        else:
            positions = x[:, :, :3]
            scales = x[:, :, 3:6]
            rotations = x[:, :, 6:10]
            if self._config.fix_mlp_order:
                opacities = x[:, :, 10: 11]
                colors = x[:, :, 11:]
            else:
                colors = x[:, :, 10:10 + self._n_color_channels]
                opacities = x[:, :, 10 + self._n_color_channels:10 + self._n_color_channels + 1]

            scales = torch.clip(scales + self._config.scale_offset, min=self._config.scale_min, max=self._config.scale_max)
            positions = self._config.position_range * positions.tanh()

        return positions, scales, rotations, colors, colors_sh, opacities

    def _upsample_feature_map(self, feature_map: torch.Tensor, perform_sampling: bool = True) -> torch.Tensor:
        if self._config.upscale_uv_texture is not None:
            if self._config.use_stylegan_pixelshuffle_upsampler:
                with torch.autocast(device_type="cuda", enabled=False):
                    feature_map = self._stylegan_upsampler(feature_map.float(), ws=None)
            else:
                feature_map = self._uv_texture_pixel_shuffle(feature_map)

        if perform_sampling:
            uv_samples = self._uv_samples.repeat(feature_map.shape[0], 1, 1, 1)  # [BV, G, 1, 2]
            sampled_features = torch.nn.functional.grid_sample(feature_map, uv_samples)[:, :, :, 0]  # [BV, D, G]
            sampled_features = sampled_features.permute(0, 2, 1)
        else:
            sampled_features = feature_map.flatten(2, 3)
            sampled_features = sampled_features.permute(0, 2, 1)

        return sampled_features

    def forward(self,
                x: torch.Tensor,
                return_uv_attributes: bool = False) -> Tuple[List[GaussianModel], Dict[str, torch.Tensor]]:

        use_mesh_gaussians = self._config.use_head_tokens
        B, HT, C = x.shape
        gaussian_predictions = dict()
        if use_mesh_gaussians:
            positions, scales, rotations, colors, colors_sh, opacities = self._decode_gaussians(self._mlp_decoder, x)
            gaussian_predictions['positions'] = positions
            initial_positions = self._initial_gaussian_positions.repeat(B, 1, 1).repeat_interleave(self._config.n_gaussians_per_token, dim=1).to(
                positions.dtype)

            positions = initial_positions + positions

        B = x.shape[0]

        gaussian_models = []
        for i in range(B):
            gaussian_model = GaussianModel(sh_degree=self._config.sh_degree)
            gaussian_model.active_sh_degree = self._config.sh_degree
            if self._config.use_lam_gs_decoder:
                gaussian_model.opacity_activation = Identity()
                gaussian_model.inverse_opacity_activation = Identity()
                gaussian_model.scaling_activation = Identity()
                gaussian_model.scaling_inverse_activation = Identity()
            gaussian_model._xyz = positions[i]
            gaussian_model._scaling = scales[i]
            gaussian_model._rotation = rotations[i]
            gaussian_model._features_dc = colors[i, :, None]
            gaussian_model._opacity = opacities[i]

            if self._config.sh_degree == 0:
                gaussian_model._features_rest = torch.empty(
                    (gaussian_model._features_dc.shape[0], 0, gaussian_model._features_dc.shape[2]), device=self.device)
            else:
                gaussian_model._features_rest = colors_sh[i]
            gaussian_models.append(gaussian_model)

        gaussian_predictions['all_positions'] = positions
        gaussian_predictions['all_scales'] = scales
        gaussian_predictions['all_rotations'] = rotations
        gaussian_predictions['all_colors'] = colors
        gaussian_predictions['all_opacities'] = opacities

        if return_uv_attributes:
            positions_mesh, scales_mesh, rotations_mesh, colors_mesh, colors_sh_mesh, opacities_mesh = self._decode_gaussians(self._mlp_decoder, x,
                                                                                                                              perform_sampling=False)
            gaussian_predictions['uv_positions'] = positions_mesh
            gaussian_predictions['uv_scales'] = scales_mesh
            gaussian_predictions['uv_rotations'] = rotations_mesh
            gaussian_predictions['uv_colors'] = colors_mesh
            gaussian_predictions['uv_opacities'] = opacities_mesh

        return gaussian_models, gaussian_predictions


class FlexAvatarModel(nn.Module):

    def __init__(self, config: FlexAvatarModelConfig):
        super().__init__()
        conv_in_channels = config.in_channels
        if config.use_plucker:
            conv_in_channels += 6
        self._conv_patchify = nn.Conv2d(in_channels=conv_in_channels, out_channels=config.head_transformer.transformer.d_hidden,
                                        kernel_size=config.patch_size,
                                        stride=config.patch_size,
                                        bias=False)

        if config.head_transformer.use_lam_transformer:
            gpt_config = GPTConfig(
                block_size=(512 // config.patch_size) ** 2 * config.n_input_views,
                n_layer=config.n_layers_encoder,
                n_head=config.head_transformer.transformer.n_heads,
                n_embd=config.head_transformer.transformer.d_hidden,
                use_post_layer_norm=True,
                n_merged_views=config.n_input_views,
                use_causal_attention=config.head_transformer.transformer.use_causal_attention,
                patch_size=config.patch_size
            )
            self._transformer_encoder = GPT(gpt_config)

        self._head_transformer = HeadTransformer(config.head_transformer)
        self._gaussian_decoder = GaussianDecoder(config.gaussian_decoder)

        if config.use_feature_projection:
            self._feature_projection = nn.Linear(config.head_transformer.transformer.d_hidden + config.feature_dim,
                                                 config.head_transformer.transformer.d_hidden)

        if config.compile and platform.system() == 'Linux':
            self.create_gaussian_models = torch.compile(self.create_gaussian_models, mode='reduce-overhead')

        self._config = config

        self.register_buffer("_device_indicator", torch.empty(0), persistent=False)

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        return super().cuda(device)

    @property
    def device(self):
        return self._device_indicator.device

    def create_gaussian_models(self,
                               images: torch.Tensor,
                               features: Optional[torch.Tensor] = None,
                               input_cam2worlds: Optional[List[List[Pose]]] = None,
                               input_intrinsics: Optional[List[List[Intrinsics]]] = None,
                               expression_codes: Optional[torch.Tensor] = None,
                               dataset_ids: Optional[torch.Tensor] = None,
                               cached_internal_representations: Optional[torch.Tensor] = None,
                               only_internal_representations: bool = False,
                               return_uv_attributes: bool = False) -> GaussianModelsOutput:

        # images is [B, V, C, H, W]
        B, V, _, H, W = images.shape
        H_p = H // self._config.patch_size
        W_p = W // self._config.patch_size

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=self._config.use_bfloat16):
            if cached_internal_representations is None:

                x = images

                if self._config.use_plucker:
                    input_plucker_embeddings = plucker_embedder(input_cam2worlds, input_intrinsics, H, W, x.device, offset=False,
                                                                use_rppc=False)
                    x = torch.cat([x, input_plucker_embeddings], dim=2)

                x = x.flatten(0, 1)
                # Use conv to patchify images into image tokens

                x = self._conv_patchify(x)  # [B*V, D, H_p, W_p]
                x = x.unflatten(0, (B, V))  # [B, V, D, H_p, W_p]
                assert x.shape[3] == H_p
                assert x.shape[4] == W_p

                def add_image_features(x: torch.Tensor):
                    features_clean = features
                    VC = features_clean.shape[1]

                    if H_p != features_clean.shape[-2] or W_p != features_clean.shape[-1]:
                        xs = torch.linspace(-1, 1, steps=W_p, device=self.device)
                        ys = torch.linspace(-1, 1, steps=H_p, device=self.device)
                        xs, ys = torch.meshgrid(xs, ys)
                        feature_grid = torch.stack([ys, xs], dim=-1)
                        feature_grid = feature_grid[None].repeat(B * VC, 1, 1, 1)
                        sampled_features = torch.nn.functional.grid_sample(features_clean.flatten(0, 1), feature_grid).unflatten(0, (B, VC))
                    else:
                        # features and patchified image actually have the same number of patches -> no grid_sample needed
                        sampled_features = features_clean

                    sampled_features = rearrange(sampled_features, 'b v c h w -> (v h w) b c')
                    x_clean = x[-VC * H_p * W_p:]
                    x_clean = torch.cat([x_clean, sampled_features], dim=2)
                    x_clean = self._feature_projection(x_clean)
                    x = torch.cat([x[:-VC * H_p * W_p], x_clean], dim=0)

                    return x

                # encode image tokens
                x = rearrange(x, 'b v c h w -> (v h w) b c')
                x = self._transformer_encoder(x)

                if self._config.use_feature_projection and features is not None:
                    x = add_image_features(x)
            else:
                x = None

            # Run cross-attention
            x, internal_representations = self._head_transformer(
                x,
                expression_codes=expression_codes,
                dataset_ids=dataset_ids,
                cached_internal_representations=cached_internal_representations,
                only_internal_representations=only_internal_representations)

            if only_internal_representations:
                return GaussianModelsOutput(None, None, internal_representations)

            x = rearrange(x, 'g b c -> b g c')

            # Decode into Gaussian Attributes

            gaussian_models, gaussian_predictions = self._gaussian_decoder(x, return_uv_attributes=return_uv_attributes)

        if self._config.use_bfloat16:
            for gaussian_model in gaussian_models:
                gaussian_model._xyz = gaussian_model._xyz.to(torch.float32)
                gaussian_model._scaling = gaussian_model._scaling.to(torch.float32)
                gaussian_model._rotation = gaussian_model._rotation.to(torch.float32)
                gaussian_model._features_dc = gaussian_model._features_dc.to(torch.float32)
                gaussian_model._features_rest = gaussian_model._features_rest.to(torch.float32)
                gaussian_model._opacity = gaussian_model._opacity.to(torch.float32)

        if expression_codes is None:
            gaussian_models_per_person = [[gaussian_model] for gaussian_model in gaussian_models]
        else:
            VT = expression_codes.shape[1]
            gaussian_models_per_person = [gaussian_models[i * VT: (i + 1) * VT] for i in range(B)]

        gaussian_models_output = GaussianModelsOutput(gaussian_models_per_person, gaussian_predictions, internal_representations=internal_representations)

        return gaussian_models_output

    def render(self, gaussian_models: List[List[GaussianModel]], batch: FlexAvatarBatch, use_gsplat: Optional[bool] = None) -> RenderingOutput:
        if isinstance(batch.render_resolution[0], int):
            img_w = batch.render_resolution[0]
            img_h = batch.render_resolution[0]
        else:
            img_w, img_h = batch.render_resolution[0]

        use_gsplat = self._config.use_gsplat if use_gsplat is None else use_gsplat

        render_bg_colors = torch.stack([torch.tensor(render_bg_color, device=self.device) / 255. for render_bg_color in batch.render_bg_color])

        rendered_images = []
        all_gaussian_models = []
        all_render_cams = []
        all_override_colors = []
        all_render_bg_colors = []
        for i, gaussian_model_list in enumerate(gaussian_models):
            rendered_images_single = []
            for v in range(len(batch.render_cam2world_poses[i])):
                if len(gaussian_model_list) == 1:
                    # Assume that same Gaussian model should be rendered from multiple views if there is only one
                    gaussian_model = gaussian_model_list[0]
                else:
                    assert len(gaussian_model_list) == len(batch.render_cam2world_poses[
                                                               i]), f"Expected #render cameras ({len(batch.render_cam2world_poses[i])}) to be the same as #gaussian models ({len(gaussian_model_list)}).)"
                    gaussian_model = gaussian_model_list[v]
                render_cam = pose_to_rendercam(batch.render_cam2world_poses[i][v], batch.render_intrinsics[i][v], img_w, img_h)
                override_color = None
                if use_gsplat:
                    all_gaussian_models.append(gaussian_model)
                    all_render_cams.append(render_cam)
                    if override_color is not None:
                        all_override_colors.append(override_color)
                    all_render_bg_colors.append(render_bg_colors[i])
                else:
                    render_output = render_distwar(render_cam, gaussian_model, PipelineParams2(), render_bg_colors[i], override_color=override_color)
                    rendered_image = render_output['render']

                    rendered_images_single.append(rendered_image)

            if not use_gsplat:
                rendered_images_single = torch.stack(rendered_images_single)
                rendered_images.append(rendered_images_single)

        if use_gsplat:
            if len(all_override_colors) == 0:
                all_override_colors = None
            render_output = render_gsplat_batched(all_render_cams, all_gaussian_models, torch.stack(all_render_bg_colors), override_color=all_override_colors)
            rendered_images = render_output["render"]
            rendered_images = rendered_images.unflatten(0, (batch.B, -1))
        else:
            rendered_images = torch.stack(rendered_images)

        output = RenderingOutput(rendered_images)

        return output

    def forward(self,
                batch: FlexAvatarBatch,
                cached_internal_representations: Optional[torch.Tensor] = None,
                only_internal_representations: bool = False,
                only_gaussian_models: bool = False,
                return_uv_attributes: bool = False) -> FlexAvatarOutput:

        gaussian_models_output = self.create_gaussian_models(batch.input_images,
                                                             features=batch.features,
                                                             input_cam2worlds=batch.input_cam2worlds,
                                                             input_intrinsics=batch.input_intrinsics,
                                                             expression_codes=batch.expression_codes,
                                                             dataset_ids=batch.dataset_ids,
                                                             cached_internal_representations=cached_internal_representations,
                                                             only_internal_representations=only_internal_representations,
                                                             return_uv_attributes=return_uv_attributes)
        rendering_output = None
        if not only_internal_representations and not only_gaussian_models:
            rendering_output = self.render(gaussian_models_output.gaussian_models, batch)

        output = FlexAvatarOutput(gaussian_models_output, rendering_output)

        return output