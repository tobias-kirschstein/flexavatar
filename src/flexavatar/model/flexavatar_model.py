import platform
from dataclasses import dataclass, replace
from typing import Optional, List, Tuple, Union, Dict

import numpy as np
import torch
from dreifus.camera import PoseType
from dreifus.matrix import Pose, Intrinsics
from dreifus.vector import Vec3
from einops import rearrange
from elias.config import Config
from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render_distwar, render_gsplat_batched
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import pose_to_rendercam
from gaussian_splatting.utils.sh_utils import C0, eval_sh

from flexavatar.config.dataset_config import DATASET_ID_MAPPING, GaussianHeadLRMBatch, MVDatasetConfig
from flexavatar.config.flexavatar_config import HeadTransformerConfig, HeadTransformerType, CrossAttentionType, TransformerConfig
from flexavatar.env import ASSETS_PATH
from flexavatar.model.flexavatar_preprocessor import GaussianHeadLRMPreprocessor
from flexavatar.model.lam_gs_layer import GSLayer
from flexavatar.model.lam_point_embedder import PointEmbed
from flexavatar.model.lam_transformer import TransformerDecoder
from flexavatar.model.nanogpt import GPTConfig, GPT
from flexavatar.model.stylegan_upsampler import StyleGANUpsamplerConfig, StyleGANPixelShuffleUpsampler
from flexavatar.util.plucker import plucker_embedder
from flexavatar.util.uv import gen_tritex
from torch import nn, device
from torch.nn import GELU, LayerNorm, PixelShuffle, Identity
from torch.nn.modules.module import T
from torchvision.ops import MLP
from trimesh import load_mesh


class Transformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._config = config

        attention_layers = []
        for _ in range(config.n_layers):
            attention_layers.append(ResidualAttentionBlock(config.d_hidden,
                                                           config.n_heads,
                                                           use_custom_attention=config.use_custom_attention,
                                                           use_qk_norm=config.use_qk_norm))
        self._attention_layers = nn.ModuleList(attention_layers)

        self_attention_layers = []
        if config.use_alternating_self_attention:
            for _ in range(config.n_layers):
                self_attention_layers.append(ResidualAttentionBlock(config.d_hidden,
                                                                    config.n_heads,
                                                                    use_custom_attention=config.use_custom_attention,
                                                                    use_qk_norm=config.use_qk_norm))
        self._self_attention_layers = nn.ModuleList(self_attention_layers)

        if config.use_layer_norm_keys:
            self._layer_norm_keys = LayerNorm(config.d_hidden)

    def forward(self, x: torch.Tensor, keys: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._config.use_layer_norm_keys and keys is not None:
            keys = self._layer_norm_keys(keys)

        if self._config.use_alternating_self_attention:
            for attention_layer, self_attention_layer in zip(self._attention_layers, self._self_attention_layers):
                x = attention_layer(x, keys=keys)
                x = self_attention_layer(x)

        else:
            for attention_layer in self._attention_layers:
                x = attention_layer(x, keys=keys)

        return x


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

        # self._head_token_embeddings = nn.Parameter(torch.zeros((config.res_head_tokens ** 2, 1, config.transformer.d_hidden)))  # [HT, 1, D]
        if config.head_transformer_type == HeadTransformerType.MESH_TOKENS:
            self._head_token_embeddings = nn.Parameter(torch.zeros((len(initial_gaussian_positions), 1, config.transformer.d_hidden)))  # [HT, 1, D]
            initial_head_xyz = initial_gaussian_positions[None]
        elif config.head_transformer_type == HeadTransformerType.UV_TEXTURE:
            self._head_token_embeddings = nn.Parameter(torch.zeros((config.res_head_tokens ** 2, 1, config.transformer.d_hidden)))  # [HT, 1, D]
            initial_head_xyz = position_map.reshape(1, config.res_head_tokens ** 2, 3)
        else:
            raise ValueError(f"Unknown head transformer type: {config.head_transformer_type}")

        if config.use_head_xyz_input or config.use_lam_point_embedder:
            self.register_buffer("_initial_head_xyz", initial_head_xyz, persistent=False)
            if config.use_head_xyz_input:
                self._head_xyz_mlp = MLP(config.transformer.d_hidden + 3, [4 * config.transformer.d_hidden, config.transformer.d_hidden],
                                         activation_layer=GELU)

            if config.use_lam_point_embedder:
                self._query_point_embedder = PointEmbed(dim=config.transformer.d_hidden)

        if config.use_image_token_embeddings:
            self._image_token_embeddings = nn.Parameter(
                torch.zeros((config.n_input_views * config.res_image_tokens ** 2, 1, config.transformer.d_hidden)))  # [VHT, 1, D]
            # nn.init.normal_(self._image_token_embeddings, mean=0.0, std=0.02)
            # nn.init.normal_(self._image_token_embeddings, mean=0.0, std=0.2)
            nn.init.trunc_normal_(self._image_token_embeddings)

        if config.use_lam_transformer:
            self._transformer = TransformerDecoder('sd3_cond', config.transformer.n_layers, config.transformer.n_heads, config.transformer.d_hidden,
                                                   cond_dim=config.transformer.d_hidden,
                                                   use_ada_ln=False,
                                                   transform_keys=False)  # TODO: cond_dim could be different
        else:
            self._transformer = Transformer(config.transformer)

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
                                                                  transform_keys=False)  # TODO: cond_dim could be different

            else:
                self._expression_transformer = Transformer(config.transformer)

        if config.d_residual_codes is not None:
            self._residual_mlp = MLP(config.d_residual_codes,
                                     [256] * 2 + [
                                         config.transformer.d_hidden if config.n_residual_tokens is None else config.transformer.d_hidden * config.n_residual_tokens],
                                     activation_layer=torch.nn.ReLU)

            self._residual_transformer = TransformerDecoder('sd3_cond',
                                                            config.transformer.n_layers, config.transformer.n_heads, config.transformer.d_hidden,
                                                            cond_dim=config.transformer.d_hidden,
                                                            use_ada_ln=False,
                                                            transform_keys=False)  # TODO: cond_dim could be different

            for layer in self._residual_transformer.layers:
                layer.ff.net[-1].weight.data[:] = 0
                layer.ff.net[-1].bias.data[:] = 0
                layer.attn.to_out[0].weight.data[:] = 0
                layer.attn.to_out[0].bias.data[:] = 0
                # torch.nn.init.zeros_(layer.ff.net[-1].weight)
                # torch.nn.init.zeros_(layer.ff.net[-1].bias)
                # torch.nn.init.zeros_(layer.attn.to_out[0].weight)
                # torch.nn.init.zeros_(layer.attn.to_out[0].bias)
            # self._residual_transformer.layers[i].ff.net[-1].weight / bias
            # self._residual_transformer.layers[i].attn.to_out[0]

        if self._config.use_dataset_ids:
            if self._config.use_separate_dataset_ids:
                n_dataset_ids = len(DATASET_ID_MAPPING)
            elif self._config.use_nersemble_dataset_ids:
                n_dataset_ids = 17
            else:
                n_dataset_ids = 2
            self._dataset_embedding = nn.Embedding(n_dataset_ids, config.transformer.d_hidden)

        if self._config.use_ln_before_transformer:
            self._query_ln = LayerNorm(config.transformer.d_hidden)
            self._input_ln = LayerNorm(config.transformer.d_hidden)

            if self._config.use_image_token_embeddings:
                self._image_token_embeddings_ln = LayerNorm(config.transformer.d_hidden)

        # nn.init.trunc_normal_(self._head_token_embeddings, std=0.02)
        nn.init.trunc_normal_(self._head_token_embeddings)

        if self._config.use_backprojected_xyz_input:
            # Maps [D + 3] -> [D]
            self._backprojected_xyz_mlp = MLP(config.transformer.d_hidden + 3, [4 * config.transformer.d_hidden, config.transformer.d_hidden],
                                              activation_layer=GELU)

            if self._config.use_image_token_embeddings:
                self._backprojected_xyz_image_token_embeddings_mlp = MLP(config.transformer.d_hidden + 3,
                                                                         [4 * config.transformer.d_hidden, config.transformer.d_hidden],
                                                                         activation_layer=GELU)

        if config.use_representation_compressor:
            self._representation_compressor = RepresentationCompressor(RepresentationCompressorConfig(config.res_head_tokens,
                                                                                                      config.n_compression_steps,
                                                                                                      config.n_layers_per_compression,
                                                                                                      config.transformer.d_hidden,
                                                                                                      config.transformer.n_heads,
                                                                                                      use_lam_point_embedder=not config.use_learnable_compression_queries,
                                                                                                      head_template=config.head_template))

    def forward(self, x: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                input_cam2worlds: Optional[List[List[Pose]]] = None,
                input_intrinsics: Optional[List[List[Intrinsics]]] = None,
                expression_codes: Optional[torch.Tensor] = None,
                residual_codes: Optional[torch.Tensor] = None,
                dataset_ids: Optional[torch.Tensor] = None,
                cached_internal_representations: Optional[torch.Tensor] = None,
                only_internal_representations: bool = False,
                ) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:

        if cached_internal_representations is None:
            B = x.shape[1]
            if not self._config.use_head_tokens:
                queries = torch.empty((0, B, self._config.transformer.d_hidden), device=x.device, dtype=x.dtype)
            elif self._config.use_lam_point_embedder:
                queries = self._query_point_embedder(self._initial_head_xyz).permute(1, 0, 2).repeat(1, B, 1).to(x.dtype)
            else:
                queries = self._head_token_embeddings.repeat(1, B, 1).to(x.dtype)
            pixel_aligned_predictions = None

            backprojected_xyz = None
            if self._config.use_backprojected_xyz_input:
                res_input = self._config.res_image_tokens  # int(sqrt(x.shape[0]))  # TODO: Only works with single input image
                backprojected_xyz = get_unprojected_points(input_cam2worlds, input_intrinsics, res_input, x.dtype, x.device)
                backprojected_xyz = backprojected_xyz.permute(1, 0, 2)

                x = x + self._backprojected_xyz_mlp(torch.cat([x, backprojected_xyz], dim=-1))

            if self._config.use_head_xyz_input:
                initial_head_xyz = self._initial_head_xyz.repeat(B, 1, 1).permute(1, 0, 2)
                queries = queries + self._head_xyz_mlp(torch.cat([queries, initial_head_xyz], dim=-1))

            # p = pv.Plotter()
            # p.add_points(initial_head_xyz[:, 0].float().detach().cpu().numpy(), color='red')
            # p.add_points(backprojected_xyz[:, 0].float().detach().cpu().numpy(), color='blue')
            # for pose, intr in zip(input_cam2worlds[0], input_intrinsics[0]):
            #     add_camera_frustum(p, pose, intr.rescale(128, inplace=False))
            # add_coordinate_axes(p, scale=0.1)
            # p.show()

            if self._config.use_ln_before_transformer:
                queries = self._query_ln(queries)
                x = self._input_ln(x)

            if condition is not None and len(condition.shape) == 3:
                # TODO: Decoder cannot really make use of timestep condition of multiple input images. Hence, only using the condition of the first timestep here
                condition = condition[:, 0]

            if self._config.cross_attention_type == CrossAttentionType.Q2K:
                if self._config.use_image_token_embeddings:
                    image_queries = self._image_token_embeddings.repeat(1, B, 1).to(x.dtype)
                    if self._config.use_backprojected_xyz_input:
                        image_queries = image_queries + self._backprojected_xyz_image_token_embeddings_mlp(
                            torch.cat([image_queries, backprojected_xyz], dim=-1))
                    if self._config.use_ln_before_transformer:
                        image_queries = self._image_token_embeddings_ln(image_queries)
                    qi = torch.cat([queries, image_queries], dim=0)
                    x = self._transformer(qi, keys=x, condition=condition)
                    pixel_aligned_predictions = x[-len(image_queries):]
                    x = x[:len(queries)]
                else:
                    x = self._transformer(queries, keys=x, condition=condition)
            elif self._config.cross_attention_type == CrossAttentionType.Q2QK:
                qk = torch.cat([queries, x], dim=0)
                x = self._transformer(queries, keys=qk, condition=condition)
            elif self._config.cross_attention_type == CrossAttentionType.QK2QK:
                if self._config.use_image_token_embeddings:
                    image_queries = self._image_token_embeddings.repeat(1, B, 1).to(x.dtype)
                    if self._config.use_backprojected_xyz_input:
                        image_queries = image_queries + self._backprojected_xyz_image_token_embeddings_mlp(
                            torch.cat([image_queries, backprojected_xyz], dim=-1))
                    if self._config.use_ln_before_transformer:
                        image_queries = self._image_token_embeddings_ln(image_queries)
                    qki = torch.cat([queries, x, image_queries], dim=0)
                    x = self._transformer(qki, condition=condition)
                    pixel_aligned_predictions = x[-len(image_queries):]
                else:
                    qk = torch.cat([queries, x], dim=0)
                    x = self._transformer(qk, condition=condition)
                    pixel_aligned_predictions = x[len(queries):]
                x = x[:len(queries)]
            else:
                raise ValueError(f"Unknown cross attention type: {self._config.cross_attention_type}")

            if self._config.use_representation_compressor:
                x = self._representation_compressor.compress(x)
        else:
            x = cached_internal_representations[:self._head_token_embeddings.shape[0]]
            pixel_aligned_predictions = cached_internal_representations[self._head_token_embeddings.shape[0]:]
            if pixel_aligned_predictions.shape[0] == 0:
                pixel_aligned_predictions = None

        if pixel_aligned_predictions is not None:
            x = torch.cat([x, pixel_aligned_predictions])

        vae_output = None

        internal_representation = x

        if only_internal_representations:
            return x, pixel_aligned_predictions, internal_representation, vae_output

        if self._config.use_representation_compressor:
            x = self._representation_compressor.decompress(x)

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
            if condition is not None:
                condition = condition.repeat_interleave(expression_codes.shape[1], dim=0)

            if self._config.use_dataset_ids:
                if self._config.use_nersemble_dataset_ids and dataset_ids[0, 0] == 9999:
                    # Average over NeRSemble dataset IDs
                    dataset_tokens = self._dataset_embedding(torch.arange(16, device=x.device)).mean(dim=0)
                    dataset_tokens = dataset_tokens[None, None]
                else:
                    dataset_tokens = self._dataset_embedding(dataset_ids)
                if self._config.use_nersemble_dataset_ids:
                    # In case of NeRSemble dataset IDs ablation, no repeat is necessary, since the IDs are per target image and not per sample
                    dataset_tokens = dataset_tokens.flatten(0, 1)[None]  # [1, B, D]
                else:
                    dataset_tokens = dataset_tokens.permute(1, 0, 2)  # [1, B, D]
                    dataset_tokens = dataset_tokens.repeat_interleave(expression_codes.shape[1], dim=1)
                expression_tokens = torch.cat([dataset_tokens, expression_tokens], dim=0)

        if self._config.d_expression_codes:
            x = self._expression_transformer(x, keys=expression_tokens, condition=condition)
            if pixel_aligned_predictions is not None:
                pixel_aligned_predictions = x[-len(pixel_aligned_predictions):]
                x = x[:-len(pixel_aligned_predictions)]

        if self._config.d_residual_codes is not None:
            B = x.shape[1]

            if pixel_aligned_predictions is not None:
                x = torch.cat([x, pixel_aligned_predictions])

            if self._config.n_residual_tokens is None:
                residual_tokens = self._residual_mlp(residual_codes).flatten(0, 1)
            else:
                residual_tokens = self._residual_mlp(residual_codes).reshape(B,
                                                                             self._config.n_residual_tokens,
                                                                             self._config.transformer.d_hidden)
            residual_tokens = residual_tokens.permute(1, 0, 2)

            x = self._residual_transformer(x, keys=residual_tokens, condition=condition)
            if pixel_aligned_predictions is not None:
                pixel_aligned_predictions = x[-len(pixel_aligned_predictions):]
                x = x[:-len(pixel_aligned_predictions)]

        return x, pixel_aligned_predictions, internal_representation, vae_output


@dataclass
class RepresentationCompressorConfig(Config):
    resolution: int
    n_compression_steps: int
    n_layers_per_compression: int
    d_hidden: int
    n_heads: int
    use_lam_point_embedder: bool
    head_template: str = 'gghead_template'


class CompressionLayer(nn.Module):
    def __init__(self, config: RepresentationCompressorConfig, resolution: int):
        super().__init__()
        self._config = config

        if config.use_lam_point_embedder:
            _, _, position_map = sample_template_positions(resolution, config.head_template)
            position_map = position_map.reshape(1, resolution ** 2, 3)
            self._query_point_embedder = PointEmbed(dim=config.d_hidden)
            self.register_buffer('_position_map', position_map, persistent=False)
        else:
            self._learnable_queries = nn.Parameter(torch.zeros((resolution ** 2, 1, config.d_hidden)))
            nn.init.trunc_normal_(self._learnable_queries)

        self._transformer = TransformerDecoder('sd3_cond', config.n_layers_per_compression, config.n_heads, config.d_hidden,
                                               cond_dim=config.d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._config.use_lam_point_embedder:
            queries = self._query_point_embedder(self._position_map).permute(1, 0, 2).repeat(1, x.shape[1], 1).to(x.dtype)
        else:
            queries = self._learnable_queries.repeat(1, x.shape[1], 1).to(x.dtype)

        x = self._transformer(queries, x)
        return x


class RepresentationCompressor(nn.Module):
    def __init__(self, config: RepresentationCompressorConfig):
        super().__init__()
        self._config = config

        compression_layers = []
        decompression_layers = []

        resolution = config.resolution
        for i_step in range(config.n_compression_steps):
            lower_res = resolution // 2
            compression_layers.append(CompressionLayer(config, lower_res))
            resolution = lower_res

        for i_step in range(config.n_compression_steps):
            higher_res = resolution * 2
            decompression_layers.append(CompressionLayer(config, higher_res))
            resolution = higher_res

        self._compression_layers = nn.ModuleList(compression_layers)
        self._decompression_layers = nn.ModuleList(decompression_layers)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        # Downsampling
        for i_step in range(self._config.n_compression_steps):
            x = self._compression_layers[i_step](x)

        return x

    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        # Upsampling
        for i_step in range(self._config.n_compression_steps):
            x = self._decompression_layers[i_step](x)

        return x


def unproject_depth(depth_map, fxfycxcy, c2w):
    """
    Unproject depth to 3D space (world coordinate).
    """
    B, N, H, W = depth_map.shape
    depth_map = depth_map.reshape(B * N, H, W)
    # fxfycxcy = fxfycxcy.reshape(-1, fxfycxcy.shape[-1])
    K = fxfycxcy.view(B * N, 3, 3)
    # K = torch.zeros(B * N, 3, 3, device=depth_map.device)
    # K[:, 0, 0] = fxfycxcy[:, 0]
    # K[:, 1, 1] = fxfycxcy[:, 1]
    # K[:, 0, 2] = fxfycxcy[:, 2]
    # K[:, 1, 2] = fxfycxcy[:, 3]
    # K[:, 2, 2] = 1
    c2w = c2w.reshape(B * N, 4, 4)
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    y = y.to(depth_map.device).unsqueeze(0).repeat(B * N, 1, 1) / (H - 1)
    x = x.to(depth_map.device).unsqueeze(0).repeat(B * N, 1, 1) / (W - 1)
    xy_map = torch.stack([x, y], axis=-1) * depth_map[..., None]
    xyz_map = torch.cat([xy_map, depth_map[..., None]], axis=-1)
    xyz = xyz_map.view(B * N, -1, 3)

    # get point positions in camera coordinate
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    K_inv = torch.eye(3, device=depth_map.device, dtype=depth_map.dtype).repeat(B * N, 1, 1)
    K_inv[..., 0, 0] = 1 / fx
    K_inv[..., 1, 1] = 1 / fy
    K_inv[..., 0, 2] = -1 * cx / fx
    K_inv[..., 1, 2] = -1 * cy / fy
    xyz = torch.matmul(xyz, torch.transpose(K_inv, -1, -2))
    xyz_map = xyz.view(B * N, H, W, 3)

    # transform pts from camera to world coordinate
    xyz_homo = torch.ones((B * N, H, W, 4), device=depth_map.device)
    xyz_homo[..., :3] = xyz_map
    xyz_world = torch.bmm(c2w, xyz_homo.reshape(B * N, -1, 4).permute(0, 2, 1)).permute(
        0, 2, 1)[..., :3].reshape(B, N * H * W, 3)
    return xyz_world


def get_unprojected_points(input_cam2worlds: Optional[List[List[Pose]]],
                           input_intrinsics: Optional[List[List[Intrinsics]]],
                           resolution: int,
                           dtype: torch.dtype,
                           device: torch.device,
                           predicted_depth: Optional[torch.Tensor] = None):
    B = len(input_cam2worlds)
    input_c2ws = torch.tensor(np.stack(input_cam2worlds), dtype=dtype, device=device)
    input_intr = torch.tensor(np.stack(input_intrinsics), dtype=dtype, device=device)
    V = input_c2ws.shape[1]
    ray_o = input_c2ws[:, :, :3, 3]
    res_aligned = resolution
    depth_offset = torch.norm(ray_o, dim=-1, p=2, keepdim=True)
    depth_init = depth_offset.repeat(1, 1, res_aligned ** 2)
    depth_init = depth_init.reshape(B, V, res_aligned, res_aligned)

    if predicted_depth is not None:
        VT = predicted_depth.shape[0] // B
        depth_init = depth_init.repeat_interleave(VT, dim=0) + predicted_depth.reshape(B * VT, V, res_aligned,
                                                                                       res_aligned)  # TODO: Probably won't work with more n_gaussians_per_token
        # If we created separate 3D representations for each target view (e.g., due to expression change), need to repeat input poses/intrinsics for each output view
        input_intr = input_intr.repeat_interleave(VT, dim=0)
        input_c2ws = input_c2ws.repeat_interleave(VT, dim=0)

    xyz_init = unproject_depth(depth_init, input_intr, input_c2ws)

    return xyz_init


@dataclass
class GaussianDecoderConfig(Config):
    n_gaussians_per_token: int
    res_head_tokens: int
    d_hidden: int
    n_mlp_layers: int
    scale_offset: float
    scale_max: float
    scale_min: float = -40
    position_range: float = 0.4
    res_uv_texture: int = 256
    res_image_tokens: Optional[int] = None
    upscale_uv_texture: Optional[int] = None
    head_transformer_type: HeadTransformerType = HeadTransformerType.MESH_TOKENS
    use_stylegan_pixelshuffle_upsampler: bool = False
    sample_aligned_gaussians: bool = False
    use_norm_before_mlp: bool = True
    initialize_with_image: bool = False
    n_channels_color: int = 3
    use_variance_channels: bool = False
    fix_mlp_order: bool = False
    use_color_skip: bool = False
    init_zero_color: bool = False
    init_zero_variance: bool = False
    use_variance_activation: bool = False
    predict_depth: bool = False
    use_separate_mlp_decoder: bool = False
    use_head_tokens: bool = True
    oversampling_factor: int = 1  # By how much generated (uv)-textures should be oversampled to spawn Gaussians
    use_lam_gs_decoder: bool = False
    use_gaussians: bool = True  # If False, simply decode 1 pixel value per pixel_aligned_token
    use_vae: bool = False
    make_contiguous: bool = False
    sh_degree: int = 0
    head_template: str = 'gghead_template'
    d_expression_codes: Optional[int] = None


class GaussianDecoder(nn.Module):
    def __init__(self, config: GaussianDecoderConfig):
        super().__init__()
        self._config = config

        self._n_color_channels = 2 * config.n_channels_color if config.use_variance_channels else config.n_channels_color

        d_feature_maps = config.d_hidden
        mlp_d_in = d_feature_maps
        if config.head_transformer_type == HeadTransformerType.UV_TEXTURE and config.upscale_uv_texture is not None:
            mlp_d_in = mlp_d_in // config.upscale_uv_texture ** 2
            assert mlp_d_in * config.upscale_uv_texture ** 2 == d_feature_maps, "MLP input size needs to be divisible by upscale factor"

        if config.use_norm_before_mlp:
            self._layer_norm = LayerNorm(mlp_d_in)

        if config.use_gaussians:
            self._mlp_decoder = self.create_mlp_decoder(config, mlp_d_in)
        else:
            self._mlp_decoder = MLP(mlp_d_in, [config.d_hidden] * (config.n_mlp_layers - 1) + [config.n_channels_color],
                                    activation_layer=GELU)
            if config.use_vae:
                self._out_activation = nn.Identity()
            else:
                self._out_activation = nn.Sigmoid()

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

        if config.use_gaussians:
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
            if not config.init_zero_color:
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
            x = self._layer_norm(x)  # TODO: Should we use layer norm here?
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

        if self._config.use_variance_channels:
            variances = colors[:, :, self._config.n_channels_color:]
            if self._config.use_variance_activation:
                # Force variances into [0, 1]
                variances = variances.sigmoid()

            colors = torch.cat([colors[:, :, :self._config.n_channels_color], variances], dim=-1)

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
                pixel_aligned_predictions: Optional[torch.Tensor] = None,
                input_cam2worlds: Optional[List[List[Pose]]] = None,
                input_intrinsics: Optional[List[List[Intrinsics]]] = None,
                input_images: Optional[torch.Tensor] = None,
                only_mesh_gaussians: bool = False,
                only_pixel_aligned_gaussians: bool = False,
                return_uv_attributes: bool = False,
                expression_codes: Optional[torch.Tensor] = None) -> Tuple[List[GaussianModel], Dict[str, torch.Tensor]]:

        if self._config.use_gaussians:
            use_mesh_gaussians = self._config.use_head_tokens and not only_pixel_aligned_gaussians
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
                if self._config.make_contiguous:
                    gaussian_model._xyz = positions[i].contiguous()
                    gaussian_model._scaling = scales[i].contiguous()
                    gaussian_model._rotation = rotations[i].contiguous()
                    gaussian_model._features_dc = colors[i, :, None].contiguous()
                    gaussian_model._opacity = opacities[i].contiguous()
                else:
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

        else:
            b, v, c, h, w = pixel_aligned_predictions.shape
            pixel_aligned_predictions = rearrange(pixel_aligned_predictions, 'b v c h w -> (b v) c h w')
            pixel_aligned_predictions = self._upsample_feature_map(pixel_aligned_predictions,
                                                                   aligned=True,
                                                                   perform_sampling=self._config.sample_aligned_gaussians)  # [B*E, V*H*W, D]
            pixel_aligned_predictions = self._mlp_decoder(pixel_aligned_predictions)
            pixel_aligned_predictions = self._out_activation(pixel_aligned_predictions)
            gaussian_models = rearrange(pixel_aligned_predictions, '(b v) (h w) c -> b v c h w', v=v,
                                        h=h * self._config.upscale_uv_texture * self._config.oversampling_factor,
                                        w=w * self._config.upscale_uv_texture * self._config.oversampling_factor)
            gaussian_predictions = dict()

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


@dataclass
class GaussianHeadLRMConfig(Config):
    head_transformer: HeadTransformerConfig
    gaussian_decoder: GaussianDecoderConfig
    patch_size: int
    in_channels: int
    n_layers_encoder: int
    n_input_views: int = 1
    use_feature_projection: bool = False
    add_features_before_encoder: bool = False
    feature_dim: int = 1536
    use_bfloat16: bool = False
    no_color_clamp: bool = False
    compute_headpose_sh: bool = False
    normalize_images: bool = False
    encode_images_separately: bool = False
    residual_downsample: int = 4  # Ensure target images cannot leak too much by forcing downsampling
    n_layers_residual_encoder: int = 4
    use_plucker: bool = False
    use_rppc: bool = False  # Reference-Point Plucker Coordinates

    compile: bool = False
    use_gsplat: bool = False


@dataclass
class RenderingOutput:
    rendered_images: torch.Tensor  # [B, TV, C, H, W]
    rendered_raw_images: Optional[torch.Tensor] = None


@dataclass
class GaussianModelsOutput:
    gaussian_models: List[List[GaussianModel]]
    gaussian_predictions: Optional[Dict[str, torch.Tensor]] = None
    x_repa: Optional[torch.Tensor] = None
    internal_representations: Optional[torch.Tensor] = None
    residual_codes: Optional[torch.Tensor] = None
    vae_output: Optional = None


@dataclass
class GaussianHeadLRMOutput:
    gaussian_models_output: GaussianModelsOutput
    rendering_output: RenderingOutput


class GaussianHeadLRM(nn.Module):

    def __init__(self, config: GaussianHeadLRMConfig):
        super().__init__()
        conv_in_channels = config.in_channels
        if config.use_plucker:
            conv_in_channels += 6
        self._conv_patchify = nn.Conv2d(in_channels=conv_in_channels, out_channels=config.head_transformer.transformer.d_hidden,
                                        kernel_size=config.patch_size,
                                        stride=config.patch_size,
                                        bias=False)

        if config.head_transformer.use_lam_transformer:
            # TODO: We are also using a GPT transformer encoder even when we use a LAM TransformerDecoder
            gpt_config = GPTConfig(
                block_size=(512 // config.patch_size) ** 2 * config.n_input_views,  # TODO: Here we assume input images will be 512x512 resolution
                n_layer=config.n_layers_encoder,
                n_head=config.head_transformer.transformer.n_heads,
                n_embd=config.head_transformer.transformer.d_hidden,
                use_adaptive_layer_norm=False,
                init_adaptive_layer_norm_identity=False,
                use_repa=config.head_transformer.use_repa,
                repa_layer=config.head_transformer.repa_layer,
                d_repa_target=config.head_transformer.d_repa_target,
                use_post_layer_norm=config.head_transformer.use_transformer_encoder_ln,
                n_merged_views=1 if config.encode_images_separately else config.n_input_views,
                use_causal_attention=config.head_transformer.transformer.use_causal_attention,
                use_prope=False,
                patch_size=config.patch_size
            )
            self._transformer_encoder = GPT(gpt_config)
        else:
            transformer_encoder_config = replace(config.head_transformer.transformer, n_layers=config.n_layers_encoder, use_alternating_self_attention=False)
            self._transformer_encoder = Transformer(transformer_encoder_config)

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

    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
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
                               residual_codes: Optional[torch.Tensor] = None,
                               dataset_ids: Optional[torch.Tensor] = None,
                               condition: Optional[torch.Tensor] = None,
                               input_view_mask: Optional[torch.Tensor] = None,
                               cached_internal_representations: Optional[torch.Tensor] = None,
                               only_mesh_gaussians: bool = False,
                               only_pixel_aligned_gaussians: bool = False,
                               only_internal_representations: bool = False,
                               return_uv_attributes: bool = False) -> GaussianModelsOutput:

        # images is [B, V, C, H, W]
        B, V, _, H, W = images.shape
        H_p = H // self._config.patch_size
        W_p = W // self._config.patch_size

        x_repa = None

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=self._config.use_bfloat16):
            if cached_internal_representations is None:

                x = images

                if self._config.use_plucker:
                    input_plucker_embeddings = plucker_embedder(input_cam2worlds, input_intrinsics, H, W, x.device, offset=False,
                                                                use_rppc=self._config.use_rppc)
                    x = torch.cat([x, input_plucker_embeddings], dim=2)

                x = x.flatten(0, 1)
                # x = images[:, 0]  # TODO: Currently assuming single input image
                # Use conv to patchify images into image tokens

                x = self._conv_patchify(x)  # [B*V, D, H_p, W_p]
                x = x.unflatten(0, (B, V))  # [B, V, D, H_p, W_p]
                assert x.shape[3] == H_p
                assert x.shape[4] == W_p

                def add_image_features(x: torch.Tensor):
                    VC = (input_view_mask[0] == 1).sum().item() if input_view_mask is not None else 0
                    if features.shape[1] == V and VC > 0:
                        # Only use features of clean input views, if features for all views were provided
                        features_clean = features[:, -VC:]
                    else:
                        features_clean = features

                    VC = features_clean.shape[1]

                    if H_p != features_clean.shape[-2] or W_p != features_clean.shape[-1]:
                        # TODO: Currently, features also assume just a single input image
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

                if self._config.use_feature_projection and self._config.add_features_before_encoder and features is not None:
                    x = rearrange(x, 'b v c h w -> (v h w) b c')
                    x = add_image_features(x)
                    x = rearrange(x, '(v h w) b c -> b v c h w', v=V, h=H_p, w=W_p)

                # encode image tokens
                xs = [x]
                conditions = [condition]

                prope_poses = None
                prope_intrinsics = None

                if self._config.encode_images_separately:
                    xs = [rearrange(x, 'b v c h w -> (h w) (b v) c') for x in xs]
                    conditions_encoder = [condition.flatten(0, 1) if len(condition.shape) == 3 else condition for condition in conditions]
                else:
                    xs = [rearrange(x, 'b v c h w -> (v h w) b c') for x in xs]
                    conditions_encoder = conditions

                if self._config.head_transformer.use_repa:
                    x, x_repa = self._transformer_encoder(xs[0], condition=conditions_encoder[0], poses=prope_poses, intrinsics=prope_intrinsics)
                else:
                    x = self._transformer_encoder(xs[0], condition=conditions_encoder[0], poses=prope_poses, intrinsics=prope_intrinsics)

                if self._config.encode_images_separately:
                    x = rearrange(x, '(h w) (b v) c -> (v h w) b c', h=H_p, w=W_p, b=B, v=V)
                    if x_repa is not None:
                        x_repa = rearrange(x_repa, '(b v) (h w) c -> b (v h w) c', h=H_p, w=W_p, b=B, v=V)

                if self._config.use_feature_projection and not self._config.add_features_before_encoder and features is not None:
                    x = add_image_features(x)
            else:
                x = None

            # Run cross-attention
            x, pixel_aligned_predictions, internal_representations, vae_output = self._head_transformer(
                x, condition=condition, input_cam2worlds=input_cam2worlds,
                input_intrinsics=input_intrinsics,
                expression_codes=expression_codes,
                residual_codes=residual_codes,
                dataset_ids=dataset_ids,
                cached_internal_representations=cached_internal_representations,
                only_internal_representations=only_internal_representations)

            if only_internal_representations:
                return GaussianModelsOutput(None, None, x_repa, internal_representations, vae_output)

            x = rearrange(x, 'g b c -> b g c')
            if pixel_aligned_predictions is not None:
                pixel_aligned_predictions = rearrange(pixel_aligned_predictions, '(v h w) b c -> b v c h w', v=V, h=H_p, w=W_p)

            # Decode into Gaussian Attributes

            gaussian_models, gaussian_predictions = self._gaussian_decoder(
                x, pixel_aligned_predictions,
                input_cam2worlds=input_cam2worlds, input_intrinsics=input_intrinsics, input_images=images,
                only_mesh_gaussians=only_mesh_gaussians, only_pixel_aligned_gaussians=only_pixel_aligned_gaussians,
                return_uv_attributes=return_uv_attributes, expression_codes=expression_codes)

        if self._config.gaussian_decoder.use_gaussians:
            if self._config.use_bfloat16:
                for gaussian_model in gaussian_models:
                    gaussian_model._xyz = gaussian_model._xyz.to(torch.float32)
                    gaussian_model._scaling = gaussian_model._scaling.to(torch.float32)
                    gaussian_model._rotation = gaussian_model._rotation.to(torch.float32)
                    gaussian_model._features_dc = gaussian_model._features_dc.to(torch.float32)
                    gaussian_model._features_rest = gaussian_model._features_rest.to(torch.float32)
                    gaussian_model._opacity = gaussian_model._opacity.to(torch.float32)
            # bfloat16_caster.__exit__(None, None, None)

            gaussian_models_per_person = []  # Each person can have multiple gaussian models for the individual expressions
            if expression_codes is None:
                gaussian_models_per_person = [[gaussian_model] for gaussian_model in gaussian_models]
            else:
                VT = expression_codes.shape[1]
                gaussian_models_per_person = [gaussian_models[i * VT: (i + 1) * VT] for i in range(B)]

            gaussian_models_output = GaussianModelsOutput(gaussian_models_per_person, gaussian_predictions, x_repa,
                                                          internal_representations=internal_representations, vae_output=vae_output)
        else:
            gaussian_models_output = GaussianModelsOutput(gaussian_models, gaussian_predictions, x_repa,
                                                          internal_representations=internal_representations, vae_output=vae_output)

        return gaussian_models_output

    def render(self, gaussian_models: List[List[GaussianModel]], batch: GaussianHeadLRMBatch, use_gsplat: Optional[bool] = None) -> RenderingOutput:
        if not self._config.gaussian_decoder.use_gaussians:
            return RenderingOutput(gaussian_models)

        if isinstance(batch.render_resolution[0], int):
            img_w = batch.render_resolution[0]
            img_h = batch.render_resolution[0]
        else:
            img_w, img_h = batch.render_resolution[0]

        use_gsplat = self._config.use_gsplat if use_gsplat is None else use_gsplat

        render_bg_colors = torch.stack([torch.tensor(render_bg_color, device=self.device) / 255. for render_bg_color in batch.render_bg_color])
        if self._config.gaussian_decoder.use_variance_channels:
            if self._config.gaussian_decoder.init_zero_variance:
                render_bg_colors = torch.cat([render_bg_colors, torch.zeros_like(render_bg_colors)], dim=-1)
            else:
                render_bg_colors = torch.cat([render_bg_colors, -1 * torch.ones_like(render_bg_colors)],
                                             dim=-1)  # Background has fix the smallest possible variance

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
                if self._config.no_color_clamp or self._config.gaussian_decoder.use_variance_channels:
                    override_color = gaussian_model._features_dc[:, 0]
                    if self._config.gaussian_decoder.use_variance_channels and not self._config.no_color_clamp:
                        override_color = torch.cat([
                            torch.clamp_min(override_color[..., :self._config.gaussian_decoder.n_channels_color] + 0.5, 0.0),
                            override_color[..., self._config.gaussian_decoder.n_channels_color:]], dim=1)
                        # override_color[..., :self._config.gaussian_decoder.n_channels_color] = torch.clamp_min(
                        #     override_color[..., :self._config.gaussian_decoder.n_channels_color] + 0.5, 0.0)
                elif self._config.compute_headpose_sh:
                    world2model_pose = batch.render_head_poses[i][v].invert()
                    world_location = torch.tensor(world2model_pose.get_translation(), device=self.device)
                    C = gaussian_model._features_dc.shape[2]
                    shs_view = gaussian_model.get_features.transpose(1, 2).view(-1, C, (self._config.gaussian_decoder.sh_degree + 1) ** 2)
                    dir_pp = gaussian_model.get_xyz - world_location[None]
                    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(self._config.gaussian_decoder.sh_degree, shs_view, dir_pp_normalized)
                    override_color = torch.clamp_min(sh2rgb + 0.5, 0.0)

                if use_gsplat:
                    all_gaussian_models.append(gaussian_model)
                    all_render_cams.append(render_cam)
                    if override_color is not None:
                        all_override_colors.append(override_color)
                    all_render_bg_colors.append(render_bg_colors[i])
                else:
                    render_output = render_distwar(render_cam, gaussian_model, PipelineParams2(), render_bg_colors[i], override_color=override_color)
                    # render_output = render_gsplat(render_cam, gaussian_model, render_bg_color, override_color=override_color)
                    # render_output = render(render_cam, gaussian_model, PipelineParams2(convert_SHs_python=override_color is None), render_bg_color,
                    #                        override_color=override_color)  # TamingGS apparently has some issue with back-propagating color when SHs are computed in CUDA...

                    rendered_image = render_output['render']

                    if self._config.normalize_images:
                        # [0, 1] -> [-1, 1]
                        rendered_image[:self._config.gaussian_decoder.n_channels_color] = rendered_image[
                                                                                          :self._config.gaussian_decoder.n_channels_color] * 2 - 1

                    if self._config.gaussian_decoder.use_variance_channels and self._config.gaussian_decoder.init_zero_variance:
                        # Variance prediction
                        # [0, 1] -> [-1, 1]
                        rendered_image[self._config.gaussian_decoder.n_channels_color:] = rendered_image[
                                                                                          self._config.gaussian_decoder.n_channels_color:] * 2 - 1

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

    def render_uv(self, gaussian_models: List[List[GaussianModel]], batch: GaussianHeadLRMBatch, include_transparent_gaussians: bool = False) -> torch.Tensor:
        # previous_colors = []
        gaussian_models_uv = []

        def create_random_cameras(n: int):
            random_cam2worlds = []
            for _ in range(n):
                distance = 0.8
                angle_range_x = np.pi
                angle_range_y = np.pi / 8
                random_angle_x = angle_range_x * (np.random.random() * 2 - 1)
                random_angle_y = angle_range_y * (np.random.random() * 2 - 1)
                random_x = np.sin(random_angle_x) * distance
                random_y = np.sin(random_angle_y) * distance
                random_z = np.sqrt(distance - random_x ** 2 - random_y ** 2)
                if np.random.random() > 0.5:
                    # Camera look from the back of the head
                    random_z = -random_z
                # random_x, random_y = np.random.random(2) * 2 - 1
                random_cam2world = Pose(pose_type=PoseType.CAM_2_WORLD)
                random_cam2world.move(random_x, random_y, random_z)
                random_cam2world.look_at(Vec3(), up=Vec3(0, 1, 0))
                random_cam2worlds.append(random_cam2world)

            return random_cam2worlds

        for b in range(len(gaussian_models)):
            single_previous_colors = []
            single_gaussian_models_uv = []
            for v in range(len(gaussian_models[b])):
                # single_previous_colors.append(gaussian_models[b][v]._features_dc)
                uv_colors = self._gaussian_decoder._uv_samples.squeeze(0)  # [G, 2]
                uv_colors = torch.concatenate(
                    [uv_colors, torch.zeros((uv_colors.shape[0], 1, 1), device=self.device), -torch.ones((uv_colors.shape[0], 1, 1), device=self.device)],
                    dim=-1)  # [G, 4]
                uv_colors = (uv_colors - 0.5) / C0
                # gaussian_models[b][v]._features_dc = uv_colors

                gaussian_model = gaussian_models[b][v]
                gm = GaussianModel(0)
                gm._xyz = torch.cat([gaussian_model._xyz, torch.tensor([[0, 0, -0.07]], device=self.device)])
                gm._scaling = torch.cat([gaussian_model._scaling, torch.tensor([[0.08, 0.05, 0.04]], device=self.device) / 2])
                gm._rotation = torch.cat([gaussian_model._rotation, torch.ones((1, 4), device=self.device)])
                gm._features_dc = torch.cat([uv_colors, (torch.tensor([[[0, 0, 1, -1]]], device=self.device) - 0.5) / C0])
                gm._features_rest = torch.empty((gm._xyz.shape[0], 0, 4), device=self.device)
                gm._opacity = torch.cat([gaussian_model._opacity, torch.ones((1, 1), device=self.device)])
                gm.scaling_activation = gaussian_model.scaling_activation
                gm.opacity_activation = gaussian_model.opacity_activation
                single_gaussian_models_uv.append(gm)

            gaussian_models_uv.append(single_gaussian_models_uv)
            # previous_colors.append(single_previous_colors)

            # if include_transparent_gaussians:
            #     debug_gaussian_attributes[GaussianAttribute.OPACITY] = torch.ones_like(
            #         debug_gaussian_attributes[GaussianAttribute.OPACITY])

        # ellipsoid_mesh = gaussians_to_mesh(torch.tensor([[0, 0, -0.05]]), torch.tensor([[0.08, 0.05, 0.05]]), torch.ones((1, 4)), torch.zeros((1, 3)),
        #                                    torch.ones((1, 1)))

        batch_uv = replace(batch, render_bg_color=[(255, 255, 255, 255) for _ in range(batch.B)],
                           render_cam2world_poses=[create_random_cameras(batch.VT) for b in range(batch.B)])
        rendering_output = self.render(gaussian_models_uv, batch_uv, use_gsplat=False)

        # for b in range(len(gaussian_models)):
        #     for v in range(len(gaussian_models[b])):
        #         gaussian_models[b][v]._features_dc = previous_colors[b][v]

        return rendering_output.rendered_images

    def forward_residual_encoder(self, images: torch.Tensor) -> torch.Tensor:
        return None

    def forward(self,
                batch: GaussianHeadLRMBatch,
                condition: Optional[torch.Tensor] = None,
                cached_internal_representations: Optional[torch.Tensor] = None,
                only_internal_representations: bool = False,
                only_gaussian_models: bool = False,
                return_uv_attributes: bool = False) -> GaussianHeadLRMOutput:

        if batch.residual_codes is None:
            residual_codes = self.forward_residual_encoder(batch.target_images)
        else:
            residual_codes = batch.residual_codes

        gaussian_models_output = self.create_gaussian_models(batch.input_images,
                                                             features=batch.features,
                                                             input_cam2worlds=batch.input_cam2worlds,
                                                             input_intrinsics=batch.input_intrinsics,
                                                             condition=condition,
                                                             input_view_mask=batch.input_view_mask,
                                                             expression_codes=batch.expression_codes,
                                                             residual_codes=residual_codes,
                                                             dataset_ids=batch.dataset_ids,
                                                             cached_internal_representations=cached_internal_representations,
                                                             only_internal_representations=only_internal_representations,
                                                             return_uv_attributes=return_uv_attributes)
        rendering_output = None
        if not only_internal_representations and not only_gaussian_models:
            rendering_output = self.render(gaussian_models_output.gaussian_models, batch)

        gaussian_models_output.residual_codes = residual_codes
        output = GaussianHeadLRMOutput(gaussian_models_output, rendering_output)

        return output


if __name__ == '__main__':
    from elias.util.io import load_json
    from flexavatar.data_adapter.in_the_wild_data_adapter import InTheWildDataAdapter
    from flexavatar.config.dataset_config import SampleMetadata
    from visage.matting.modnet import MODNetMatter

    data_adapter = InTheWildDataAdapter("tobi")
    sample_metadata = SampleMetadata("tobi", None, 0, None)
    image = data_adapter.load_image(sample_metadata)

    canonical_flame_to_world, _ = data_adapter.load_head_pose(sample_metadata)
    cam2world_pose, intrinsics = data_adapter.load_camera_params(sample_metadata)

    flame2world_pose = Pose(
        canonical_flame_to_world.invert().numpy() @ cam2world_pose,
        pose_type=PoseType.CAM_2_WORLD)

    device = torch.device('cuda')
    image_torch = torch.tensor(image / 255, dtype=torch.float32).permute(2, 0, 1)[None]
    modnet_matter = MODNetMatter()
    with torch.no_grad():
        alpha_maps = modnet_matter.parse(image_torch).cpu()
    image_torch = image_torch * alpha_maps[:, None] + 1 - alpha_maps[:, None]

    expression_code = torch.zeros((1, 1, 135))
    expression_code[:, :, :126] = torch.tensor(data_adapter.load_expression_code(sample_metadata))
    batch = GaussianHeadLRMBatch(image_torch[:, None], None, [[flame2world_pose]], [[intrinsics.rescale(1 / 512, inplace=False)]], None, None, None, None, None, None,
                                 expression_codes=expression_code,
                                 dataset_ids=torch.ones((1, 1), dtype=torch.long))
    batch = batch.to(device)

    model_folder = "D:/Projects/PhD-7_Photoreal_3DMM/code_release/models/SLRM-1522"
    dataset_config = MVDatasetConfig.from_json(load_json(f"{model_folder}/dataset_config.json"))

    preprocessor = GaussianHeadLRMPreprocessor(dataset_config)
    batch = preprocessor.process(batch)

    model_config = GaussianHeadLRMConfig.from_json(load_json(f"{model_folder}/model_config.json"))
    model_config.use_bfloat16 = False
    model = GaussianHeadLRM(model_config)

    checkpoint = torch.load(f"{model_folder}/checkpoints/ckpt-1050k.pt")
    model.load_state_dict(checkpoint)
    model.to(device)
    with torch.no_grad():
        output = model.create_gaussian_models(batch.input_images,
                                              batch.features,
                                              batch.input_cam2worlds,
                                              batch.input_intrinsics,
                                              expression_codes=batch.expression_codes,
                                              dataset_ids=batch.dataset_ids)

        resolution = 512
        render_cam = pose_to_rendercam(flame2world_pose, intrinsics, resolution, resolution)
        rendering_output = render_distwar(render_cam, output.gaussian_models[0][0], PipelineParams2(), torch.ones((3,), device=device))
        rendered_image = rendering_output['render'].permute(1, 2, 0).detach().cpu().numpy()

    print('hi')
