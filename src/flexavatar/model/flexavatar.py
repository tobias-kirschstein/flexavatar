import platform
from dataclasses import dataclass, replace
from enum import auto
from math import sqrt
from typing import Optional, List, Tuple, Union, Dict

import numpy as np
import torch
from dreifus.camera import PoseType
from dreifus.graphics import Dimensions
from dreifus.matrix import Pose, Intrinsics
from dreifus.render import project
from dreifus.vector import Vec3
from einops import rearrange
from elias.config import Config, StringEnum
from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render_distwar, render_gsplat_batched
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import pose_to_rendercam
from gaussian_splatting.utils.sh_utils import C0, eval_sh
from photoreal_3dmm.dataset.config import InputType, GaussianHeadLRMBatch, RenderType, DATASET_ID_MAPPING
from photoreal_3dmm.env import ASSETS_PATH
from photoreal_3dmm.model.DPR.DPR import DPR
from photoreal_3dmm.model.cgs.point_generator import PointGenerator, PointGeneratorConfig
from photoreal_3dmm.model.cnn import CNNDecoder
from photoreal_3dmm.model.dit import TimestepEmbedder, DiT
from photoreal_3dmm.model.flame.flame_deformer import FlameDeformer, expression_codes_to_flame_params
from photoreal_3dmm.model.lam_gs_layer import GSLayer
from photoreal_3dmm.model.lam_point_embedder import PointEmbed
from photoreal_3dmm.model.lam_transformer import TransformerDecoder
from photoreal_3dmm.model.mlp_bundle import MLPBundle
from photoreal_3dmm.model.nanogpt_orig import GPTConfig, GPT
from photoreal_3dmm.model.res_att_block import ResidualAttentionBlock
from photoreal_3dmm.model.stylegan_upsampler import StyleGANUpsampler, StyleGANUpsamplerConfig, StyleGANPixelShuffleUpsampler
from photoreal_3dmm.model.vae import VAEModule, VAEOutput
from photoreal_3dmm.util.plucker import plucker_embedder
from photoreal_3dmm.util.uv import gen_tritex
from torch import nn, device
from torch.nn import GELU, LayerNorm, PixelShuffle, Identity
from torch.nn.functional import interpolate
from torch.nn.modules.module import T
from torchvision.ops import MLP
from trimesh import load_mesh


@dataclass
class TransformerConfig(Config):
    n_layers: int
    d_hidden: int
    n_heads: int
    use_custom_attention: bool = False
    use_qk_norm: bool = False
    use_layer_norm_keys: bool = False
    use_alternating_self_attention: bool = False
    use_causal_attention: bool = True


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


class HeadTransformerType(StringEnum):
    MESH_TOKENS = auto()
    UV_TEXTURE = auto()


class CrossAttentionType(StringEnum):
    Q2K = auto()
    Q2QK = auto()
    QK2QK = auto()


@dataclass
class HeadTransformerConfig(Config):
    transformer: TransformerConfig
    res_head_tokens: int
    head_transformer_type: HeadTransformerType = HeadTransformerType.MESH_TOKENS
    cross_attention_type: CrossAttentionType = CrossAttentionType.Q2K
    use_lam_transformer: bool = False
    use_lam_point_embedder: bool = False
    use_gpt: bool = False
    use_adaptive_layer_norm: bool = False
    init_adaptive_layer_norm_identity: bool = False
    res_image_tokens: Optional[int] = None
    n_input_views: int = 1
    use_image_token_embeddings: bool = False
    use_pixel_aligned_gaussians: bool = False
    use_repa: bool = False
    repa_layer: int = -1
    d_repa_target: int = 768
    use_backprojected_xyz_input: bool = False
    use_head_xyz_input: bool = False
    block_size_estimate_version: int = 1
    use_ln_before_transformer: bool = False
    use_transformer_encoder_ln: bool = True
    use_transformer_decoder_ln: bool = True
    d_expression_codes: Optional[int] = None
    n_expression_tokens: Optional[int] = 4
    d_residual_codes: Optional[int] = None
    n_residual_tokens: Optional[int] = None
    use_head_tokens: bool = True
    n_layers_expression_transformer: int = 4
    use_vae: bool = False
    use_point_generator: bool = False
    n_point_generator_layers: int = 6
    use_dataset_ids: bool = False
    use_separate_dataset_ids: bool = False
    use_nersemble_dataset_ids: bool = False
    head_template: str = 'gghead_template'

    use_representation_compressor: bool = False
    n_compression_steps: int = 3
    n_layers_per_compression: int = 1
    use_learnable_compression_queries: bool = False


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
                                                   use_ada_ln=config.use_adaptive_layer_norm,
                                                   transform_keys=config.use_pixel_aligned_gaussians)  # TODO: cond_dim could be different
        elif config.use_gpt:
            max_n_input_tokens = self._head_token_embeddings.shape[0]
            if config.cross_attention_type == CrossAttentionType.Q2K and config.use_image_token_embeddings:
                max_n_input_tokens += self._image_token_embeddings.shape[0]
            elif config.cross_attention_type == CrossAttentionType.Q2QK or config.cross_attention_type == CrossAttentionType.QK2QK:
                if config.block_size_estimate_version == 3:
                    max_n_input_tokens += config.n_input_views * config.res_image_tokens ** 2

                    if config.use_image_token_embeddings:
                        max_n_input_tokens += self._image_token_embeddings.shape[0]
                elif config.block_size_estimate_version == 2:
                    max_n_input_tokens += 2 * config.res_image_tokens ** 2
                else:
                    max_n_input_tokens *= 2

            gpt_config = GPTConfig(
                block_size=max_n_input_tokens,
                n_layer=config.transformer.n_layers,
                n_head=config.transformer.n_heads,
                n_embd=config.transformer.d_hidden,
                use_cross_attention=config.cross_attention_type != CrossAttentionType.QK2QK,
                use_adaptive_layer_norm=config.use_adaptive_layer_norm,
                init_adaptive_layer_norm_identity=config.init_adaptive_layer_norm_identity,
                use_post_layer_norm=config.use_transformer_decoder_ln,
                use_causal_attention=config.transformer.use_causal_attention,
            )
            self._transformer = GPT(gpt_config)
        else:
            self._transformer = Transformer(config.transformer)

        if config.d_expression_codes is not None:
            self._expression_mlp = MLP(config.d_expression_codes,
                                       [256] * 2 + [
                                           config.transformer.d_hidden if config.n_expression_tokens is None else config.transformer.d_hidden * config.n_expression_tokens],
                                       activation_layer=torch.nn.ReLU)

        if config.d_expression_codes is not None and not config.use_point_generator:
            if config.use_lam_transformer:
                self._expression_transformer = TransformerDecoder('sd3_cond',
                                                                  config.n_layers_expression_transformer, config.transformer.n_heads,
                                                                  config.transformer.d_hidden,
                                                                  cond_dim=config.transformer.d_hidden,
                                                                  use_ada_ln=config.use_adaptive_layer_norm,
                                                                  transform_keys=config.use_pixel_aligned_gaussians)  # TODO: cond_dim could be different
            elif config.use_gpt:
                expression_transformer_config = replace(gpt_config, use_cross_attention=True)
                self._expression_transformer = GPT(expression_transformer_config)
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
                                                            use_ada_ln=config.use_adaptive_layer_norm,
                                                            transform_keys=config.use_pixel_aligned_gaussians)  # TODO: cond_dim could be different

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

        if config.use_vae:
            self._vae = VAEModule(config.transformer.d_hidden)

        if config.use_point_generator:
            num_pts = 512
            random_coords = torch.randn((num_pts, 3))
            point_generator_xyz = random_coords * torch.rsqrt(torch.mean(random_coords ** 2, dim=1, keepdim=True) + 1e-8) * 0.5 * 0.6
            self._point_generator_xyz = nn.Parameter(point_generator_xyz)
            self._point_generator = PointGenerator(
                PointGeneratorConfig(w_dim=config.transformer.d_hidden, d_hidden=config.transformer.d_hidden, n_transformer=config.n_point_generator_layers))

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
                    if self._config.use_lam_transformer and self._config.use_pixel_aligned_gaussians:
                        x, pixel_aligned_predictions = self._transformer(queries, keys=x, condition=condition, return_keys=True)
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
        if self._config.use_vae:
            vae_output = self._vae(x)
            x = vae_output.x

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

        if self._config.use_point_generator:
            w = x[0]  # [B, D]
            x = self._point_generator(self._point_generator_xyz.repeat(w.shape[0], 1, 1), w[:, None], keys=expression_tokens)
        else:
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
    use_separate_mlps: bool = False
    use_stylegan_upsampler: bool = False
    use_stylegan_pixelshuffle_upsampler: bool = False
    use_pixel_aligned_gaussians: bool = False
    sample_aligned_gaussians: bool = False
    use_norm_before_mlp: bool = True
    initialize_with_image: bool = False
    n_channels_color: int = 3
    use_variance_channels: bool = False
    fix_mlp_order: bool = False
    use_color_skip: bool = False
    use_mesh_color_init: bool = False
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
    expand_feature_maps: int = 1  # By how much feature map channels should be up-projected before feature map is upscaled (and looses channels)
    make_contiguous: bool = False
    use_flame_deformer: bool = False
    use_pixel3dmm_flame_deformer: bool = False
    flame_deformer_type: str = 'post'
    sh_degree: int = 0
    head_template: str = 'gghead_template'
    use_expression_code_ws: bool = False
    d_expression_codes: Optional[int] = None


class GaussianDecoder(nn.Module):
    def __init__(self, config: GaussianDecoderConfig):
        super().__init__()
        self._config = config

        self._n_color_channels = 2 * config.n_channels_color if config.use_variance_channels else config.n_channels_color

        d_feature_maps = config.d_hidden * config.expand_feature_maps
        mlp_d_in = d_feature_maps
        if config.head_transformer_type == HeadTransformerType.UV_TEXTURE and config.upscale_uv_texture is not None and not config.use_stylegan_upsampler:
            mlp_d_in = mlp_d_in // config.upscale_uv_texture ** 2
            assert mlp_d_in * config.upscale_uv_texture ** 2 == d_feature_maps, "MLP input size needs to be divisible by upscale factor"

        if config.use_norm_before_mlp:
            self._layer_norm = LayerNorm(mlp_d_in)

        if config.use_gaussians:
            self._mlp_decoder = self.create_mlp_decoder(config, mlp_d_in)
            if config.use_pixel_aligned_gaussians:
                if config.use_separate_mlp_decoder:
                    self._mlp_decoder_pixel_aligned_gaussians = self.create_mlp_decoder(config, mlp_d_in)
                    # n_position_channels=1 if config.predict_depth else 3)  # TODO: Requires generalizing how we extract positions
                else:
                    self._mlp_decoder_pixel_aligned_gaussians = self._mlp_decoder
                    # self._mlp_decoder_pixel_aligned_gaussians = lambda x: self._mlp_decoder(x)
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
                if config.use_pixel_aligned_gaussians and (config.use_stylegan_upsampler or config.use_stylegan_pixelshuffle_upsampler):
                    stylegan_config_aligned = replace(stylegan_config,
                                                      input_res=config.res_image_tokens,
                                                      output_res=config.res_image_tokens * config.upscale_uv_texture)

                if config.use_stylegan_upsampler:
                    self._stylegan_upsampler = StyleGANUpsampler(stylegan_config)
                    if config.use_pixel_aligned_gaussians:
                        self._stylegan_upsampler_aligned = StyleGANUpsampler(stylegan_config_aligned)
                elif config.use_stylegan_pixelshuffle_upsampler:
                    self._stylegan_upsampler = StyleGANPixelShuffleUpsampler(stylegan_config)
                    if config.use_pixel_aligned_gaussians:
                        self._stylegan_upsampler_aligned = StyleGANPixelShuffleUpsampler(stylegan_config_aligned)

                    if config.use_expression_code_ws:
                        self._expression_code_ws_mlp = nn.Linear(config.d_expression_codes, stylegan_config.w_dim)
                else:
                    self._uv_texture_pixel_shuffle = PixelShuffle(config.upscale_uv_texture)

                if config.expand_feature_maps > 1:
                    self._feature_map_expander = nn.Linear(config.d_hidden, d_feature_maps)

        else:
            raise ValueError(f"Unknown head transformer type: {config.head_transformer_type}")

        if config.use_gaussians:
            initial_gaussian_positions = initial_gaussian_positions[None]  # [1, G, 3]
            uv_samples = uv_samples[None, :, None]  # [1, G, 1, 2]
            self.register_buffer("_initial_gaussian_positions", initial_gaussian_positions, persistent=False)
            self.register_buffer("_uv_samples", uv_samples, persistent=False)

        if config.use_pixel_aligned_gaussians and config.sample_aligned_gaussians:
            xs = torch.linspace(-1, 1, steps=config.res_image_tokens * config.upscale_uv_texture * config.oversampling_factor)
            ys = torch.linspace(-1, 1, steps=config.res_image_tokens * config.upscale_uv_texture * config.oversampling_factor)

            xs, ys = torch.meshgrid(xs, ys, indexing='ij')
            uv_samples_aligned = torch.stack([ys, xs], dim=-1)
            uv_samples_aligned = uv_samples_aligned.reshape(1, -1, 1, 2)  # [1, G_aligned, 1, 2]
            self.register_buffer("_uv_samples_aligned", uv_samples_aligned, persistent=False)

        self.register_buffer("_device_indicator", torch.empty(0), persistent=False)

        if config.use_flame_deformer:
            self._flame_deformer = FlameDeformer()

        if config.use_pixel3dmm_flame_deformer:
            from photoreal_3dmm.model.flame.pixel3dmm_flame_deformer import Pixel3DMMFlameDeformer
            self._flame_deformer = Pixel3DMMFlameDeformer(config.res_head_tokens)

        # G = initial_gaussian_positions.shape[1] * self._config.n_gaussians_per_token
        # self._xyz = torch.nn.Parameter(self._initial_gaussian_positions[0].repeat_interleave(self._config.n_gaussians_per_token, dim=0))
        # self._rotation = torch.nn.Parameter(torch.normal(0, 0.02, (G, 4), device=self.device))
        # self._scaling = torch.nn.Parameter(torch.clip(torch.normal(0, 0.02, (G, 3), device=self.device) + self._config.scale_offset, max=self._config.scale_max))
        # self._features_dc = torch.nn.Parameter(torch.normal(0, 0.02, (G, 1, 3), device=self.device))
        # self._features_rest = torch.nn.Parameter(torch.empty(
        #     (self._features_dc.shape[0], 0, self._features_dc.shape[2]), device=self.device))
        # self._opacity = torch.nn.Parameter(torch.normal(0, 0.02, (G, 1), device=self.device))
        #
        # self._dummy_mlp = MLP(config.d_hidden, [config.d_hidden] * (config.n_mlp_layers - 1) + [(3 + 3 + 4 + 1 + 3) * G],
        #                         activation_layer=GELU)
        # self._dummy_mlp_input = torch.nn.Parameter(torch.randn((1, config.d_hidden), device=self.device))
        # for p in self._dummy_mlp.parameters():
        #     if p.dim() > 1:
        #         nn.init.normal_(p, mean=0, std=0.02)
        #
        # self._dummy_tokens = torch.nn.Parameter(torch.randn((1, 64*64, config.d_hidden), device=self.device))

    @property
    def device(self):
        return self._device_indicator.device

    def create_mlp_decoder(self, config: GaussianDecoderConfig, mlp_d_in: int, n_position_channels: int = 3):

        if config.use_lam_gs_decoder:
            mlp_decoder = GSLayer(mlp_d_in, sh_degree=self._config.sh_degree, use_rgb=self._config.sh_degree == 0)
        else:
            if config.use_separate_mlps:
                out_channels = [n_position_channels, 3, 4, 1, self._n_color_channels]
                mlps = []
                for c in out_channels:
                    mlp = MLP(mlp_d_in, [config.d_hidden] * (config.n_mlp_layers - 1) + [c * config.n_gaussians_per_token],
                              activation_layer=GELU)
                    mlps.append(mlp)
                if config.init_zero_color:
                    for mlp in mlps:
                        for p in mlp.parameters():
                            if p.dim() > 1:
                                nn.init.normal_(p, mean=0, std=0.02)

                    nn.init.constant_(mlps[-1][-2].weight, 0)
                    nn.init.constant_(mlps[-1][-2].bias, 0)

                mlp_decoder = MLPBundle(mlps, interleave_factor=config.n_gaussians_per_token)

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
                          perform_sampling: bool = True,
                          aligned: bool = False,
                          expression_codes: Optional[torch.Tensor] = None):
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

            sampled_features = self._upsample_feature_map(uv_texture, perform_sampling=perform_sampling, aligned=aligned, expression_codes=expression_codes)
            # if self._config.upscale_uv_texture is not None:
            #     if self._config.use_stylegan_upsampler or self._config.use_stylegan_pixelshuffle_upsampler:
            #         with torch.autocast(device_type="cuda", enabled=False):
            #             if aligned:
            #                 # Upsample back-projected image tokens
            #                 uv_texture = self._stylegan_upsampler_aligned(uv_texture.float())
            #             else:
            #                 # Upsample head tokens
            #                 uv_texture = self._stylegan_upsampler(uv_texture.float())
            #     else:
            #         uv_texture = self._uv_texture_pixel_shuffle(uv_texture)
            #
            # if perform_sampling:
            #     if aligned:
            #         uv_samples = self._uv_samples_aligned.repeat(x.shape[0], 1, 1, 1)  # [BV, G_aligned, 1, 2]
            #     else:
            #         uv_samples = self._uv_samples.repeat(x.shape[0], 1, 1, 1)  # [BV, G, 1, 2]
            #
            #     sampled_features = torch.nn.functional.grid_sample(uv_texture, uv_samples)[:, :, :, 0]  # [BV, D, G]
            #     sampled_features = sampled_features.permute(0, 2, 1)
            # else:
            #     sampled_features = uv_texture.flatten(2, 3)
            #     sampled_features = sampled_features.permute(0, 2, 1)

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

    def _upsample_feature_map(self, feature_map: torch.Tensor, aligned: bool = False, perform_sampling: bool = True,
                              expression_codes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._config.expand_feature_maps > 1:
            feature_map = self._feature_map_expander(feature_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self._config.upscale_uv_texture is not None:
            if self._config.use_stylegan_upsampler or self._config.use_stylegan_pixelshuffle_upsampler:
                with torch.autocast(device_type="cuda", enabled=False):
                    if aligned:
                        # Upsample back-projected image tokens
                        feature_map = self._stylegan_upsampler_aligned(feature_map.float())
                    else:
                        ws = None
                        if self._config.use_expression_code_ws:
                            ws = self._expression_code_ws_mlp(expression_codes.flatten(0, 1))  # [B * VT, D]

                        # Upsample head tokens
                        feature_map = self._stylegan_upsampler(feature_map.float(), ws=ws)
            else:
                feature_map = self._uv_texture_pixel_shuffle(feature_map)

        if perform_sampling:
            if aligned:
                uv_samples = self._uv_samples_aligned.repeat(feature_map.shape[0], 1, 1, 1)  # [BV, G_aligned, 1, 2]
            else:
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
            use_pixel_aligned_gaussians = self._config.use_pixel_aligned_gaussians and pixel_aligned_predictions is not None and not only_mesh_gaussians

            B, HT, C = x.shape
            gaussian_predictions = dict()
            if use_mesh_gaussians:
                positions, scales, rotations, colors, colors_sh, opacities = self._decode_gaussians(self._mlp_decoder, x, expression_codes=expression_codes)
                gaussian_predictions['positions'] = positions
                initial_positions = self._initial_gaussian_positions.repeat(B, 1, 1).repeat_interleave(self._config.n_gaussians_per_token, dim=1).to(
                    positions.dtype)
                if self._config.use_flame_deformer and self._config.flame_deformer_type == 'pre':
                    posed_flame_params = expression_codes_to_flame_params(expression_codes.flatten(0, 1))
                    initial_positions, rotations = self._flame_deformer.forward(initial_positions, rotations, posed_flame_params)
                if self._config.use_pixel3dmm_flame_deformer:
                    deformed_position_map = self._flame_deformer.forward(expression_codes.flatten(0, 1))
                    initial_positions = torch.nn.functional.grid_sample(
                        deformed_position_map.permute(0, 3, 1, 2),
                        self._uv_samples.repeat(deformed_position_map.shape[0], 1, 1, 1))[:, :, :, 0].permute(0, 2, 1)

                positions = initial_positions + positions

                if self._config.use_mesh_color_init:
                    VT = B // input_images.shape[0]
                    input_resolution = input_images.shape[-1]
                    projected_points = torch.tensor(
                        [project(initial_positions[0].cpu().float(), pose_list[0], intrinsics_list[0].rescale(input_resolution, inplace=False))
                         for pose_list, intrinsics_list in zip(input_cam2worlds, input_intrinsics)], device=x.device, dtype=torch.float32)  # [B, G, 3]
                    sample_positions = (projected_points[:, :, None, :2] / input_resolution * 2 - 1)
                    extracted_colors = torch.nn.functional.grid_sample(input_images[:, 0], sample_positions)[..., 0]  # [B, 3, G]
                    extracted_colors = (extracted_colors - 0.5) / C0
                    colors = colors + extracted_colors.permute(0, 2, 1).repeat_interleave(VT, dim=0)

            if use_pixel_aligned_gaussians:
                b, v, c, h, w = pixel_aligned_predictions.shape
                pixel_aligned_predictions = rearrange(pixel_aligned_predictions, 'b v c h w -> b (v h w) c')
                positions_aligned, scales_aligned, rotations_aligned, colors_aligned, colors_sh_aligned, opacities_aligned = self._decode_gaussians(
                    self._mlp_decoder_pixel_aligned_gaussians,
                    pixel_aligned_predictions,
                    h=h,
                    v=v,
                    perform_sampling=self._config.sample_aligned_gaussians,
                    aligned=True,
                    expression_codes=expression_codes)  # [B*E, V*H*W, D]

                predicted_depth = None
                if self._config.predict_depth:
                    # TODO: Probably won't work with more n_gaussians_per_token
                    predicted_depth = positions_aligned[..., 0]

                res_aligned = int(sqrt(positions_aligned.shape[1] // self._config.n_gaussians_per_token / v))
                xyz_init = get_unprojected_points(input_cam2worlds, input_intrinsics, res_aligned, x.dtype, x.device, predicted_depth=predicted_depth)
                VT = x.shape[0] // input_images.shape[0]
                # TODO: Back-projection with multiple input images is broken at the moment!!!

                if self._config.use_color_skip:
                    xs = torch.linspace(-1, 1, steps=res_aligned, device=x.device)
                    ys = torch.linspace(-1, 1, steps=res_aligned, device=x.device)

                    xs, ys = torch.meshgrid(xs, ys, indexing='ij')
                    sampled_pixel_coords = torch.stack([ys, xs], dim=-1)
                    sampled_pixel_coords = sampled_pixel_coords.unsqueeze(0).repeat(B, 1, 1, 1)
                    sampled_input_rgb = torch.nn.functional.grid_sample(input_images[:, 0].repeat_interleave(VT, dim=0),
                                                                        sampled_pixel_coords)  # TODO: Assuming single input image
                    sampled_input_rgb = sampled_input_rgb.flatten(-2, -1).permute(0, 2, 1)  # [B, C, H, W] -> [B, H*W, C]
                    sampled_input_rgb = (sampled_input_rgb - 0.5) / C0
                    colors_aligned[..., :3] = colors_aligned[..., :3] + sampled_input_rgb

                # import pyvista as pv
                # from dreifus.pyvista import add_camera_frustum, add_coordinate_axes
                # p = pv.Plotter()
                # # p.add_points(xyz_init[0].detach().float().cpu().numpy(), color='blue')
                # p.add_points(xyz_init[0].detach().float().cpu().numpy(), scalars=colors_aligned[0].detach().float().cpu().numpy()[..., :3], rgb=True)
                # p.add_points(self._initial_gaussian_positions[0].detach().float().cpu().numpy(), color='red')
                # add_camera_frustum(p, input_cam2worlds[0][0], input_intrinsics[0][0])
                # add_coordinate_axes(p, scale=0.1)
                # p.show()

                if self._config.predict_depth:
                    positions_aligned = xyz_init
                else:
                    xyz_init = xyz_init.repeat_interleave(VT, dim=0).repeat_interleave(self._config.n_gaussians_per_token, dim=1)
                    if self._config.use_flame_deformer and self._config.flame_deformer_type == 'pre':
                        posed_flame_params = expression_codes_to_flame_params(expression_codes.flatten(0, 1))
                        xyz_init, rotations_aligned = self._flame_deformer.forward(xyz_init, rotations_aligned, posed_flame_params)
                    positions_aligned = positions_aligned + xyz_init

                # import pyvista as pv
                # from dreifus.pyvista import add_camera_frustum, add_coordinate_axes
                # p = pv.Plotter()
                # # p.add_points(xyz_init[0].detach().float().cpu().numpy(), color='blue')
                # p.add_points(positions_aligned[0].detach().float().cpu().numpy(), scalars=colors_aligned[0].detach().float().cpu().numpy()[..., :3], rgb=True)
                # p.add_points(positions[0].detach().float().cpu().numpy(), scalars=colors[0].detach().float().cpu().numpy()[..., :3], rgb=True)
                # # p.add_points(self._initial_gaussian_positions[0].detach().float().cpu().numpy(), color='red')
                # for pose, intr, input_image in zip(input_cam2worlds[0], input_intrinsics[0], input_images[0]):
                #     input_image = input_image.permute(1, 2, 0).cpu().numpy()
                #     add_camera_frustum(p, pose, intr.rescale(128, inplace=False), image=input_image)
                # add_coordinate_axes(p, scale=0.1)
                # p.show()

                if use_mesh_gaussians:
                    positions = torch.cat([positions, positions_aligned], dim=1)
                    scales = torch.cat([scales, scales_aligned], dim=1)
                    rotations = torch.cat([rotations, rotations_aligned], dim=1)
                    colors = torch.cat([colors, colors_aligned], dim=1)
                    opacities = torch.cat([opacities, opacities_aligned], dim=1)

                    if self._config.sh_degree > 0:
                        colors_sh = torch.cat([colors_sh, colors_sh_aligned], dim=1)
                else:
                    positions = positions_aligned
                    scales = scales_aligned
                    rotations = rotations_aligned
                    colors = colors_aligned
                    opacities = opacities_aligned

                    if self._config.sh_degree > 0:
                        colors_sh = colors_sh_aligned

            if self._config.use_flame_deformer and self._config.flame_deformer_type == 'post':
                posed_flame_params = expression_codes_to_flame_params(expression_codes.flatten(0, 1))
                positions, rotations = self._flame_deformer.forward(positions, rotations, posed_flame_params)

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

                # dummy_gaussian_model = GaussianModel(sh_degree=0)
                # dummy_gaussian_model._xyz = self._xyz
                # dummy_gaussian_model._scaling = self._scaling
                # dummy_gaussian_model._rotation = self._rotation
                # dummy_gaussian_model._features_dc = self._features_dc
                # dummy_gaussian_model._features_rest = self._features_rest
                # dummy_gaussian_model._opacity = self._opacity
                # gaussian_models.append(dummy_gaussian_model)

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
    use_clean_image_encoder: bool = False  # Use separate image encoder for clean conditioning views
    use_residual_encoder: bool = False
    use_dpr_lighting: bool = False
    residual_downsample: int = 4  # Ensure target images cannot leak too much by forcing downsampling
    n_layers_residual_encoder: int = 4
    use_prope: bool = False
    use_plucker: bool = False
    use_rppc: bool = False  # Reference-Point Plucker Coordinates

    use_neural_renderer: bool = False
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
    vae_output: Optional[VAEOutput] = None


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

        if config.head_transformer.use_gpt or config.head_transformer.use_lam_transformer:
            # TODO: We are also using a GPT transformer encoder even when we use a LAM TransformerDecoder
            gpt_config = GPTConfig(
                block_size=(512 // config.patch_size) ** 2 * config.n_input_views,  # TODO: Here we assume input images will be 512x512 resolution
                n_layer=config.n_layers_encoder,
                n_head=config.head_transformer.transformer.n_heads,
                n_embd=config.head_transformer.transformer.d_hidden,
                use_adaptive_layer_norm=config.head_transformer.use_adaptive_layer_norm,
                init_adaptive_layer_norm_identity=config.head_transformer.init_adaptive_layer_norm_identity,
                use_repa=config.head_transformer.use_repa,
                repa_layer=config.head_transformer.repa_layer,
                d_repa_target=config.head_transformer.d_repa_target,
                use_post_layer_norm=config.head_transformer.use_transformer_encoder_ln,
                n_merged_views=1 if config.encode_images_separately else config.n_input_views,
                use_causal_attention=config.head_transformer.transformer.use_causal_attention,
                use_prope=config.use_prope,
                patch_size=config.patch_size
            )
            self._transformer_encoder = GPT(gpt_config)
        else:
            transformer_encoder_config = replace(config.head_transformer.transformer, n_layers=config.n_layers_encoder, use_alternating_self_attention=False)
            self._transformer_encoder = Transformer(transformer_encoder_config)

        if config.use_clean_image_encoder:
            if config.head_transformer.use_gpt:
                self._transformer_encoder_clean = GPT(gpt_config)
            else:
                self._transformer_encoder_clean = Transformer(transformer_encoder_config)

        if config.use_residual_encoder:
            self._conv_patchify_residual = nn.Conv2d(in_channels=3, out_channels=config.head_transformer.transformer.d_hidden,
                                                     kernel_size=config.patch_size,
                                                     stride=config.patch_size,
                                                     bias=False)

            residual_encoder_config = replace(config.head_transformer.transformer, n_layers=config.n_layers_residual_encoder,
                                              use_alternating_self_attention=False)
            self._residual_encoder = Transformer(residual_encoder_config)

            self._residual_compression_mlp = MLP(config.head_transformer.transformer.d_hidden,
                                                 [256] * 2 + [config.head_transformer.d_residual_codes],
                                                 activation_layer=torch.nn.ReLU)

        if config.use_dpr_lighting:
            self._dpr = DPR()

            # def param_to_buffer(module):
            #     """Turns all parameters of a module into buffers."""
            #     modules = module.modules()
            #     module = next(modules)
            #     for name, param in dict(module.named_parameters(recurse=False)).items():
            #         delattr(module, name)  # Unregister parameter
            #         module.register_buffer(name, param, persistent=False)
            #     for name, param in dict(module._buffers.items()).items():
            #         delattr(module, name)  # There could be persistable buffers. Replace them with non-persistable ones
            #         module.register_buffer(name, param, persistent=False)
            #     for module in modules:
            #         param_to_buffer(module)
            #
            # param_to_buffer(self._dpr)

        self._head_transformer = HeadTransformer(config.head_transformer)
        if not config.head_transformer.use_point_generator:
            self._gaussian_decoder = GaussianDecoder(config.gaussian_decoder)

        if config.use_feature_projection:
            self._feature_projection = nn.Linear(config.head_transformer.transformer.d_hidden + config.feature_dim,
                                                 config.head_transformer.transformer.d_hidden)

        if config.use_neural_renderer:
            self._neural_renderer = CNNDecoder(32, 3)

        if config.compile and platform.system() == 'Linux':
            self.create_gaussian_models = torch.compile(self.create_gaussian_models, mode='reduce-overhead')

        self._config = config

        self.register_buffer("_device_indicator", torch.empty(0), persistent=False)

    def to(self, *args, **kwargs):
        if self._config.use_dpr_lighting:
            self._dpr.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
        if self._config.use_dpr_lighting:
            self._dpr.cuda()
        return super().cuda(device)

    # def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
    #     state_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
    #     state_dict = {k: v for k, v in state_dict.items() if '_dpr' not in k}  # Exclude purely pretrained DPR module from state dict
    #     return state_dict
    #
    # def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
    #     for k, p in self.named_parameters():
    #         if k.startswith('_dpr'):
    #             state_dict[k] = p
    #
    #     for prefix, module in self.named_modules():
    #         if prefix.startswith('_dpr'):
    #             for k, p in module._buffers.items():
    #                 if k not in module._non_persistent_buffers_set:
    #                     state_dict[f"{prefix}.{k}"] = p
    #
    #         # def recurse_buffers(module: nn.Module, prefix: str = ''):
    #         #     for k, p in module._buffers.items():
    #         #         state_dict[f"{prefix}.{k}"] = p
    #         #
    #         #     for k, child_module in module.named_modules():
    #         #         recurse_buffers(child_module, prefix=f"{prefix}.{k}")
    #         #
    #         # recurse_buffers(self._dpr, prefix='_dpr')
    #
    #
    #
    #     return super().load_state_dict(state_dict, strict, assign)

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
                if self._config.use_clean_image_encoder:
                    # TODO: We are assuming that all batch elements have the same number of noisy and clean views
                    VN = (input_view_mask[0] == 0).sum()
                    VC = (input_view_mask[0] == 1).sum()
                    xs = [x[input_view_mask == 0].unflatten(0, (B, VN)), x[input_view_mask == 1].unflatten(0, (B, VC))]
                    conditions = [condition[input_view_mask == 0].unflatten(0, (B, VN)), condition[input_view_mask == 1].unflatten(0, (B, VC))]
                else:
                    xs = [x]
                    conditions = [condition]

                if self._config.use_prope:
                    # PRoPe expects world2cam poses
                    prope_poses = torch.tensor([[pose.invert() for pose in pose_list] for pose_list in input_cam2worlds], device=x.device)
                    prope_intrinsics = torch.tensor(input_intrinsics, device=x.device)
                else:
                    prope_poses = None
                    prope_intrinsics = None

                if self._config.encode_images_separately:
                    xs = [rearrange(x, 'b v c h w -> (h w) (b v) c') for x in xs]
                    conditions_encoder = [condition.flatten(0, 1) if len(condition.shape) == 3 else condition for condition in conditions]
                else:
                    xs = [rearrange(x, 'b v c h w -> (v h w) b c') for x in xs]
                    conditions_encoder = conditions

                if self._config.use_clean_image_encoder:
                    # TODO: PRoPe poses/intrinsics are missing here
                    if self._config.head_transformer.use_repa:
                        x_noisy, x_repa_noisy = self._transformer_encoder(xs[0], condition=conditions_encoder[0])
                        x_clean, x_repa_clean = self._transformer_encoder_clean(xs[1], condition=conditions_encoder[1])
                        x_repa = torch.cat([x_repa_noisy.unflatten(0, (B, VN)), x_repa_clean.unflatten(0, (B, VC))], dim=1).flatten(0, 1)
                    else:
                        x_noisy = self._transformer_encoder(xs[0], condition=conditions_encoder[0])
                        x_clean = self._transformer_encoder_clean(xs[1], condition=conditions_encoder[1])

                    # x = torch.cat([x_noisy, x_clean], dim=1)
                    x = torch.cat([x_noisy.unflatten(1, (B, VN)), x_clean.unflatten(1, (B, VC))], dim=2).flatten(1, 2)
                else:
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

            if self._config.head_transformer.use_point_generator:
                gaussian_models = x
                gaussian_predictions = None
            else:

                if only_internal_representations:
                    return GaussianModelsOutput(None, None, x_repa, internal_representations, vae_output)

                x = rearrange(x, 'g b c -> b g c')
                if pixel_aligned_predictions is not None:
                    pixel_aligned_predictions = rearrange(pixel_aligned_predictions, '(v h w) b c -> b v c h w', v=V, h=H_p, w=W_p)

                # input_cam2worlds = torch.tensor(np.stack(input_cam2worlds), dtype=x.dtype, device=x.device) if input_cam2worlds is not None else None
                # input_intrinsics = torch.tensor(np.stack(input_intrinsics), dtype=x.dtype, device=x.device) if input_intrinsics is not None else None
                # Decode into Gaussian Attributes

                if not self._config.gaussian_decoder.use_gaussians:
                    VN = (input_view_mask[0] == 0).sum()
                    pixel_aligned_predictions = pixel_aligned_predictions[input_view_mask == InputType.NOISY].unflatten(0, (B, VN))

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

        if self._config.use_neural_renderer:
            render_bg_colors = torch.cat([render_bg_colors, torch.zeros((render_bg_colors.shape[0], 32 - len(render_bg_colors)), device=self.device)], dim=-1)

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

        if self._config.use_neural_renderer:
            B, VT, C, H, W = rendered_images.shape
            rgb_images = self._neural_renderer(rendered_images.flatten(0, 1))
            rgb_images = rgb_images.unflatten(0, (B, VT))
            output = RenderingOutput(rgb_images, rendered_raw_images=rendered_images)
        else:
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
        residual_codes = None
        if self._config.use_dpr_lighting:
            B, VT, _, _, _ = images.shape
            x = images.flatten(0, 1) * 2 - 1
            with torch.no_grad():
                sh = self._dpr.extract_lighting(x)[:, :, 0, 0]  # [B*VT, 9]
            residual_codes = sh.reshape(B, VT, 1, 9)
        elif self._config.use_residual_encoder:
            # Compute residual codes from target images
            B, VT, _, _, _ = images.shape
            x = images.flatten(0, 1)
            x = interpolate(x, scale_factor=1 / self._config.residual_downsample)
            x = self._conv_patchify_residual(x)
            x = rearrange(x, 'bv c h w -> (h w) bv c')
            residual_codes = self._residual_encoder(x)
            residual_codes = self._residual_compression_mlp(residual_codes)
            residual_codes = residual_codes.permute(1, 0, 2).unflatten(0, (B, VT))  # [B, VT, T, D_res]

        return residual_codes

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


@dataclass
class DenoisingGaussianHeadLRMConfig(Config):
    head_lrm: GaussianHeadLRMConfig
    predict_x0: bool = False
    predict_eps_derived_from_x0: bool = False
    learn_sigma: bool = True
    sigma_small: bool = False


@dataclass
class DenoisingGaussianHeadLRMOutput:
    gaussian_models: List[List[GaussianModel]]  # [B, VO/1]
    diffusion_output: torch.Tensor  # [B, VO, C, H, W]
    nv_output: Optional[torch.Tensor] = None  # [B, VNV, C, H, W]
    x_repa: Optional[torch.Tensor] = None
    internal_representations: Optional[torch.Tensor] = None


class DenoisingGaussianHeadLRM(nn.Module):
    def __init__(self, config: DenoisingGaussianHeadLRMConfig):
        super().__init__()
        self._t_embedder = TimestepEmbedder(config.head_lrm.head_transformer.transformer.d_hidden)

        if not config.learn_sigma or config.head_lrm.gaussian_decoder.use_variance_channels:
            head_lrm_config = config.head_lrm
        else:
            # We need to predict mean and variance => simply double output channels
            gaussian_decoder_config = replace(config.head_lrm.gaussian_decoder, n_channels_color=2 * config.head_lrm.gaussian_decoder.n_channels_color)
            head_lrm_config = replace(config.head_lrm, gaussian_decoder=gaussian_decoder_config)

        self._lifting_module = GaussianHeadLRM(head_lrm_config)

        self.reset_cache()
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize timestep embedding MLP:
        nn.init.normal_(self._t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self._t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, batch: GaussianHeadLRMBatch, cached_internal_representations: Optional[torch.Tensor] = None) -> DenoisingGaussianHeadLRMOutput:
        t_all = torch.zeros((batch.B, batch.VI), device=batch.device, dtype=t.dtype)
        t_flat = t.flatten(0, 1) if len(t.shape) == 2 else t
        t_all[batch.input_view_mask == InputType.NOISY] = t_flat

        t_all = self._t_embedder(t_all)
        x_t = batch.input_images.clone()
        x_t[batch.input_view_mask == InputType.NOISY] = x.flatten(0, 1)
        batch.input_images = x_t  # TODO: Is it ok, to overwrite here?

        output = self._lifting_module.forward(batch, condition=t_all, cached_internal_representations=cached_internal_representations)

        if batch.render_view_mask is not None:
            diffusion_output = output.rendering_output.rendered_images[batch.render_view_mask == RenderType.DIFFUSION].unflatten(0, (batch.B, -1))
            nv_output = output.rendering_output.rendered_images[batch.render_view_mask == RenderType.NV].unflatten(0, (batch.B, -1))
        else:
            diffusion_output = output.rendering_output.rendered_images
            nv_output = None

        output = DenoisingGaussianHeadLRMOutput(
            gaussian_models=output.gaussian_models_output.gaussian_models,
            diffusion_output=diffusion_output,
            nv_output=nv_output,
            x_repa=output.gaussian_models_output.x_repa,
            internal_representations=output.gaussian_models_output.internal_representations
        )

        return output

    def set_cache_trajectory(self, poses: List[Pose], intrinsics: Intrinsics):
        self._cache_poses = poses
        self._cache_intrinsics = intrinsics

    def set_condition_images(self, condition_images: torch.Tensor):
        self._condition_images = condition_images

    def set_denoising_cameras(self, denoising_poses: List[Pose], denoising_intrinsics: List[Intrinsics]):
        self._denoising_poses = denoising_poses
        self._denoising_intrinsics = denoising_intrinsics

    def set_denoising_expression_codes(self, expression_codes: List[torch.Tensor]):
        self._denoising_expression_codes = expression_codes

    def render(self, gaussian_models: List[GaussianModel], batch: GaussianHeadLRMBatch) -> RenderingOutput:
        return self._lifting_module.render(gaussian_models, batch)

    def forward_and_cache(self, x, t, batch: GaussianHeadLRMBatch):
        t_all = torch.zeros((batch.B, batch.VI), device=batch.device, dtype=t.dtype)
        t_flat = t.flatten(0, 1) if len(t.shape) == 2 else t
        t_all[batch.input_view_mask == InputType.NOISY] = t_flat

        t_all = self._t_embedder(t_all)
        x_t = batch.input_images.clone()
        x_t[batch.input_view_mask == InputType.NOISY] = x.flatten(0, 1)
        batch.input_images = x_t  # TODO: Is it ok, to overwrite here?

        if len(self._denoising_poses) > 0:
            pose = self._denoising_poses.pop(0)
            intrinsics = self._denoising_intrinsics.pop(0)
            render_cam2worlds = [[pose] for _ in range(len(batch.input_images))]
            render_intrinsics = [[intrinsics] for _ in range(len(batch.input_images))]
            batch.render_intrinsics = [[intr.rescale(x.shape[-1], x.shape[-2], inplace=False) for intr in intrinsics] for intrinsics in render_intrinsics]
            batch.render_cam2world_poses = render_cam2worlds
            input_cam2worlds = [render_cam2world + input_cam2world[len(render_cam2world):] for render_cam2world, input_cam2world in
                                zip(render_cam2worlds, batch.input_cam2worlds)]
            input_intrinsics = [render_intr + input_intr[len(render_intr):] for render_intr, input_intr in zip(render_intrinsics, batch.input_intrinsics)]
        else:
            input_cam2worlds = batch.input_cam2worlds
            input_intrinsics = batch.input_intrinsics

        if len(self._denoising_expression_codes) > 0:
            expression_code = self._denoising_expression_codes.pop(0)
            expression_codes = torch.stack([expression_code for _ in range(len(batch.input_images))])
            batch.expression_codes = expression_codes

        gaussian_models_output = self._lifting_module.create_gaussian_models(batch.input_images,
                                                                             features=batch.features,
                                                                             input_cam2worlds=input_cam2worlds,
                                                                             input_intrinsics=input_intrinsics,
                                                                             condition=t_all,
                                                                             input_view_mask=batch.input_view_mask,
                                                                             expression_codes=batch.expression_codes)
        gaussian_models = gaussian_models_output.gaussian_models
        rendered_images = self._lifting_module.render(gaussian_models, batch).rendered_images

        if self._condition_images is not None:
            B, _, _, H, W = batch.input_images.shape
            rendered_images = torch.cat([self._condition_images[:, :, :, :, :int(W / 2)], rendered_images[:, :, :, :, int(W / 2):]], dim=4)

        self._cached_gaussian_models = gaussian_models
        self._cached_internal_representations = gaussian_models_output.internal_representations
        self._denoising_history.append((x.detach().cpu() * 255).clamp(0, 255).to(torch.uint8))
        self._prediction_history.append((rendered_images.detach().cpu() * 255).clamp(0, 255).to(torch.uint8))

        if len(self._cache_poses) > 0:
            pose = self._cache_poses.pop(0)
            pose = [[pose] for _ in range(len(batch.input_images))]
            intrinsics = [[self._cache_intrinsics] for _ in range(len(batch.input_images))]
            render_resolution = [Dimensions(512, 512) for _ in range(len(batch.input_images))]
            batch_cache_trajectory = replace(batch, render_cam2world_poses=pose, render_intrinsics=intrinsics, render_resolution=render_resolution)
            rendered_images_cache = self._lifting_module.render(gaussian_models, batch_cache_trajectory).rendered_images
            self._cache_trajectory.append((rendered_images_cache[:, 0].detach().cpu() * 255).clamp(0, 255).to(torch.uint8))

        output = DenoisingGaussianHeadLRMOutput(
            gaussian_models=gaussian_models,
            diffusion_output=rendered_images,
        )

        return output

    def reset_cache(self):
        self._denoising_history = []
        self._prediction_history = []
        self._cache_trajectory = []
        self._cache_poses = []
        self._cache_intrinsics = None
        self._condition_images = None
        self._denoising_poses = []
        self._denoising_intrinsics = []
        self._denoising_expression_codes = []
        self._cached_internal_representations = None


class DenoisingDiT(DiT):
    def forward(self, x, t, batch: GaussianHeadLRMBatch, cached_internal_representations: Optional[torch.Tensor] = None) -> DenoisingGaussianHeadLRMOutput:
        diffusion_output = super().forward(x[:, 0], t[:, 0], y=torch.tensor([0], device=x.device))

        output = DenoisingGaussianHeadLRMOutput(
            gaussian_models=None,
            diffusion_output=diffusion_output[:, None],
            nv_output=None,
            x_repa=None,
            internal_representations=None
        )

        return output

    @classmethod
    def from_dit(cls, dit: DiT):
        denoising_dit = cls()
        denoising_dit.__dict__ = dit.__dict__
        return denoising_dit


@dataclass
class HeadLVSMConfig(Config):
    in_channels: int
    patch_size: int
    n_layers_encoder: int
    n_layers: int
    transformer: TransformerConfig
    n_color_channels: int = 3
    n_mlp_layers: int = 2
    use_adaptive_layer_norm: bool = False
    init_adaptive_layer_norm_identity: bool = False
    use_repa: bool = False
    repa_layer: int = -1
    d_repa_target: int = 768
    use_bfloat16: bool = False
    cross_attention_type: CrossAttentionType = CrossAttentionType.Q2K
    use_camera_transformer: bool = False
    use_positional_embedding_render: bool = True
    n_registers: int = 0
    n_registers_render: int = 0
    use_separate_renderer: bool = False  # Only relevant for diffusion training: separate renderers for decoding noise/x0 from internal representation
    use_plucker: bool = False
    use_lam_transformer: bool = False
    d_expression_codes: Optional[int] = None
    n_expression_tokens: Optional[int] = 4
    target_views_are_noise: bool = False  # If true, this is a multi-view image diffusion setting where the target views are the noisy input views
    use_vae: bool = False


@dataclass
class DenoisingHeadLVSMConfig(Config):
    head_lvsm: HeadLVSMConfig
    predict_x0: bool = False
    predict_eps_derived_from_x0: bool = False
    learn_sigma: bool = True
    sigma_small: bool = False


class HeadLVSM(nn.Module):
    def __init__(self, config: HeadLVSMConfig):
        super().__init__()
        self._config = config

        self._conv_patchify = nn.Conv2d(in_channels=config.in_channels + 6 if config.use_plucker or config.target_views_are_noise else config.in_channels,
                                        out_channels=config.transformer.d_hidden,
                                        kernel_size=config.patch_size,
                                        stride=config.patch_size,
                                        bias=False)

        self._conv_patchify_target = nn.Conv2d(in_channels=6, out_channels=config.transformer.d_hidden,
                                               kernel_size=config.patch_size,
                                               stride=config.patch_size,
                                               bias=False)

        gpt_config = GPTConfig(
            block_size=(512 // config.patch_size) ** 2,  # TODO: Here we assume input images will be 512x512 resolution
            n_layer=config.n_layers_encoder,
            n_head=config.transformer.n_heads,
            n_embd=config.transformer.d_hidden,
            use_adaptive_layer_norm=config.use_adaptive_layer_norm,
            init_adaptive_layer_norm_identity=config.init_adaptive_layer_norm_identity,
            use_repa=config.use_repa,
            repa_layer=config.repa_layer,
            d_repa_target=config.d_repa_target,
            n_registers=config.n_registers,
            use_causal_attention=config.transformer.use_causal_attention,
        )
        self._transformer_encoder = GPT(gpt_config)

        if config.use_lam_transformer:
            self._transformer_render = TransformerDecoder('sd3_cond', config.transformer.n_layers, config.transformer.n_heads,
                                                          config.transformer.d_hidden,
                                                          cond_dim=config.transformer.d_hidden,
                                                          use_ada_ln=config.use_adaptive_layer_norm,
                                                          transform_keys=False)

            if config.d_expression_codes is not None:
                self._expression_mlp = MLP(config.d_expression_codes,
                                           [256] * 2 + [
                                               config.transformer.d_hidden if config.n_expression_tokens is None else config.transformer.d_hidden * config.n_expression_tokens],
                                           activation_layer=torch.nn.ReLU)

                self._expression_transformer = TransformerDecoder('sd3_cond',
                                                                  config.transformer.n_layers, config.transformer.n_heads, config.transformer.d_hidden,
                                                                  cond_dim=config.transformer.d_hidden,
                                                                  use_ada_ln=config.use_adaptive_layer_norm,
                                                                  transform_keys=False)
        else:
            gpt_config_2 = replace(gpt_config,
                                   n_layer=config.n_layers,
                                   use_repa=False
                                   )
            # self._transformer = GPT(gpt_config_2)

            gpt_config_render = replace(gpt_config_2,
                                        block_size=(512 // config.patch_size) ** 2 + (512 // config.patch_size) * (
                                                744 // config.patch_size) * 6,  # 1 input + 1 render view
                                        use_positional_embedding=config.use_positional_embedding_render,
                                        use_cross_attention=config.cross_attention_type != CrossAttentionType.QK2QK,
                                        n_registers=config.n_registers_render,
                                        )

            self._transformer_render = GPT(gpt_config_render)

        if config.use_separate_renderer and not config.target_views_are_noise:
            self._transformer_render_separate = GPT(gpt_config_render)

        if config.use_camera_transformer:
            gpt_config_camera = replace(gpt_config,
                                        use_repa=False,
                                        use_positional_embedding=config.use_positional_embedding_render,
                                        n_registers=config.n_registers_render, )
            self._transformer_camera = GPT(gpt_config_camera)

        self._pixel_shuffle = PixelShuffle(config.patch_size)
        mlp_d_in = config.transformer.d_hidden // (config.patch_size ** 2)
        self._mlp_decoder = MLP(mlp_d_in, [config.transformer.d_hidden] * (config.n_mlp_layers - 1) + [config.n_color_channels],
                                activation_layer=GELU)

        if config.use_vae:
            self._out_activation = nn.Identity()
        else:
            self._out_activation = nn.Sigmoid()

        # if not config.learn_sigma or config.head_lrm.gaussian_decoder.use_variance_channels:
        #     head_lrm_config = config.head_lrm
        # else:
        #     # We need to predict mean and variance => simply double output channels
        #     gaussian_decoder_config = replace(config.head_lrm.gaussian_decoder, n_channels_color=2 * config.head_lrm.gaussian_decoder.n_channels_color)
        #     head_lrm_config = replace(config.head_lrm, gaussian_decoder=gaussian_decoder_config)
        #
        # self._lifting_module = GaussianHeadLRM(head_lrm_config)

        # self.reset_cache()

    def forward(self, batch: GaussianHeadLRMBatch, condition: Optional[torch.Tensor] = None,
                return_internal_representations: bool = False, return_repa: bool = False):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=self._config.use_bfloat16):

            x, x_repa = self.create_internal_representations(batch.input_images,
                                                             input_cam2worlds=batch.input_cam2worlds, input_intrinsics=batch.input_intrinsics,
                                                             condition=condition, return_repa=True)

            # target_tokens = x[:, B:]
            # x = x[:, :B]

            T_input = x.shape[0]
            # x = torch.cat([x, target_tokens], dim=0)  # Now put camera embeddings into token dimension --> dense self-attention

            # TODO: cross-view transformer not needed for now
            # x = self._transformer(x, condition=t)
            # target_features = x[T_input:]  # [HW, B, C]

            output_rgb = self.render(x, batch, t=condition).rendered_images

        if self._config.use_bfloat16:
            x = x.to(torch.float32)
            output_rgb = output_rgb.to(torch.float32)
            if x_repa is not None:
                x_repa = x_repa.to(torch.float32)

        if return_repa:
            if return_internal_representations:
                return output_rgb, x_repa, x
            else:
                return output_rgb, x_repa
        else:
            if return_internal_representations:
                return output_rgb, x
            else:
                return output_rgb

    def create_internal_representations(self,
                                        input_images: torch.Tensor,
                                        input_cam2worlds: List[List[Pose]] = None,
                                        input_intrinsics: List[List[Intrinsics]] = None,
                                        condition: Optional[torch.Tensor] = None,
                                        return_repa: bool = False):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=self._config.use_bfloat16):

            B, V, _, H, W = input_images.shape

            # TODO: input plucker

            x = input_images

            if self._config.use_plucker:
                input_plucker_embeddings = plucker_embedder(input_cam2worlds, input_intrinsics, H, W, x.device, offset=False)
                x = torch.cat([x, input_plucker_embeddings], dim=2)

            x = x.flatten(0, 1)

            # Use conv to patchify images into image tokens
            x = self._conv_patchify(x)  # [B*V, D, H_p, W_p]

            # encode image tokens
            x = rearrange(x, 'bv c h w -> (h w) bv c')
            if condition is not None:
                condition = rearrange(condition, 'b v c -> (b v) c')

            x_repa = None
            if self._config.use_repa:
                # x, x_repa = self._transformer_encoder(x, condition=t.repeat((2, 1)))  # TODO: Currently, we feed target plucker through transformer encoder and have to repeat timestep embedding
                # x_repa = x_repa[:B]
                x, x_repa = self._transformer_encoder(x, condition=condition)

                if x_repa is not None:
                    x_repa = rearrange(x_repa, '(b v) hw c -> b (v hw) c', b=B, v=V)
            else:
                x = self._transformer_encoder(x, condition=condition)

            if return_repa:
                return x, x_repa
            else:
                return x

    def render(self,
               internal_representations: torch.Tensor,
               batch: GaussianHeadLRMBatch,
               t: Optional[torch.Tensor] = None,
               use_separate_renderer: bool = False) -> RenderingOutput:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=self._config.use_bfloat16):

            B, V, _, _, _ = batch.input_images.shape
            device = internal_representations.device

            VT = len(batch.render_cam2world_poses[0])

            H_p = int(sqrt(internal_representations.shape[0]))
            W_p = H_p

            if self._config.target_views_are_noise:
                internal_representations = rearrange(internal_representations, 'hw (b v) c -> b v hw c', b=B, v=V)
                target_tokens = internal_representations[batch.input_view_mask == InputType.NOISY]
                internal_representations = internal_representations[
                    batch.input_view_mask == InputType.CLEAN]  # TODO: Have a flag for this? This only works in the conditional case now
                target_tokens = rearrange(target_tokens, 'bvt hw c -> hw bvt c')
                # internal_representations = rearrange(internal_representations, 'bvc hw c -> hw bvc c')
                internal_representations = rearrange(internal_representations, '(b vc) hw c -> (vc hw) b c', b=B)

            else:
                W, H = batch.render_resolution[0]
                render_intrinsics = [[intr.rescale(1 / W, 1 / H, inplace=False) for intr in intr_list] for intr_list in batch.render_intrinsics]
                target_camera_feature = plucker_embedder(batch.render_cam2world_poses, render_intrinsics, H, W, device)
                target_tokens = self._conv_patchify_target(target_camera_feature.flatten(0, 1))
                _, D, H_p, W_p = target_tokens.shape
                target_tokens = target_tokens.flatten(2, 3).permute(2, 0, 1)

            if self._config.use_camera_transformer:
                camera_condition = None if t is None else t.repeat_interleave(VT, dim=0)
                target_tokens = self._transformer_camera(target_tokens, condition=camera_condition)

            T_input = internal_representations.shape[0]

            if self._config.use_separate_renderer and use_separate_renderer:
                transformer_render = self._transformer_render_separate
            else:
                transformer_render = self._transformer_render

            if self._config.cross_attention_type == CrossAttentionType.Q2K:
                # TODO: This is handling each target view separately. Maybe we do not want to do that
                hw = target_tokens.shape[0]
                target_tokens = rearrange(target_tokens, 'hw (b v) c -> (v hw) b c', b=B)
                x = transformer_render(target_tokens, keys=internal_representations, condition=t)
                x = rearrange(x, '(v hw) b c -> hw (b v) c', hw=hw)
            elif self._config.cross_attention_type == CrossAttentionType.Q2QK:
                internal_representations = internal_representations.repeat_interleave(VT, dim=1)  # Copy internal representation for each target view
                qk = torch.cat([target_tokens, internal_representations], dim=0)
                x = transformer_render(target_tokens, keys=qk, condition=t)
            elif self._config.cross_attention_type == CrossAttentionType.QK2QK:
                internal_representations = internal_representations.repeat_interleave(VT, dim=1)  # Copy internal representation for each target view
                qk = torch.cat([target_tokens, internal_representations], dim=0)
                x = transformer_render(qk, condition=t)
                x = x[:len(target_tokens)]

            if self._config.d_expression_codes is not None:
                expression_codes = batch.expression_codes
                # if self._config.target_views_are_noise:
                #     expression_codes = expression_codes[batch.input_view_mask == InputType.NOISY][:, None]  # TODO: Implicitly we are assuming here that there is only one expression per batch?

                B = x.shape[1]
                if self._config.n_expression_tokens is None:
                    expression_tokens = self._expression_mlp(expression_codes).flatten(0, 1)
                else:
                    expression_tokens = self._expression_mlp(expression_codes).reshape(B * expression_codes.shape[1],
                                                                                       self._config.n_expression_tokens,
                                                                                       self._config.transformer.d_hidden)
                expression_tokens = expression_tokens.permute(1, 0, 2)

                internal_representation = x
                # Duplicate internal 3D representation for each expression code -> there will be separate GaussianModels for each expression code
                x = x.repeat_interleave(expression_codes.shape[1], dim=1)
                if t is not None:
                    condition = t.repeat_interleave(expression_codes.shape[1], dim=0)
                x = self._expression_transformer(x, keys=expression_tokens, condition=condition)

            target_features = x

            # x = torch.cat([internal_representations, target_tokens], dim=0)  # Now put camera embeddings into token dimension --> dense self-attention
            # x = self._transformer_render(x, condition=t.repeat_interleave(VT, dim=0))
            # target_features = x[T_input:]  # [HW, BV, C]

            # Upsampling
            output = rearrange(target_features, '(h w) (b v) d -> b v d h w', b=B, h=H_p, w=W_p)
            # output = target_features.reshape(H_p, W_p, B, VT, D).permute(2, 3, 4, 0, 1)  # [B, V, C, H, W]
            output = self._pixel_shuffle(output)
            output_rgb = self._mlp_decoder(output.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            output_rgb = self._out_activation(output_rgb)

        if self._config.use_bfloat16:
            output_rgb = output_rgb.to(torch.float32)

        # output_rgb = output_rgb.permute(0, 3, 1, 2)  # [B, 3, H, W]

        output = RenderingOutput(output_rgb)

        return output


class DenoisingHeadLVSM(nn.Module):

    def __init__(self, config: DenoisingHeadLVSMConfig):
        super().__init__()
        self._t_embedder = TimestepEmbedder(config.head_lvsm.transformer.d_hidden)

        self._lifting_module = HeadLVSM(config.head_lvsm)

        self.reset_cache()
        self.initialize_weights()
        self._config = config

    def initialize_weights(self):
        # Initialize timestep embedding MLP:
        nn.init.normal_(self._t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self._t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, batch: GaussianHeadLRMBatch):
        t_all = torch.zeros((batch.B, batch.VI), device=batch.device, dtype=t.dtype)
        t_flat = t.flatten(0, 1) if len(t.shape) == 2 else t
        t_all[batch.input_view_mask == InputType.NOISY] = t_flat

        t_all = self._t_embedder(t_all)
        x_t = batch.input_images.clone()
        x_t[batch.input_view_mask == InputType.NOISY] = x.flatten(0, 1)
        batch.input_images = x_t  # TODO: Is it ok, to overwrite here?

        output, x_repa, internal_representations = self._lifting_module.forward(batch, condition=t_all, return_internal_representations=True, return_repa=True)

        output = DenoisingGaussianHeadLRMOutput(
            gaussian_models=internal_representations,
            diffusion_output=output,
            x_repa=x_repa
        )

        return output

    def forward_and_cache(self, x, t, batch: GaussianHeadLRMBatch):
        t_all = torch.zeros((batch.B, batch.VI), device=batch.device, dtype=t.dtype)
        t_flat = t.flatten(0, 1) if len(t.shape) == 2 else t
        t_all[batch.input_view_mask == InputType.NOISY] = t_flat

        t_all = self._t_embedder(t_all)
        x_t = batch.input_images.clone()
        x_t[batch.input_view_mask == InputType.NOISY] = x.flatten(0, 1)
        batch.input_images = x_t  # TODO: Is it ok, to overwrite here?

        internal_representations = self._lifting_module.create_internal_representations(batch.input_images,
                                                                                        input_cam2worlds=batch.input_cam2worlds,
                                                                                        input_intrinsics=batch.input_intrinsics,
                                                                                        condition=t_all)
        rendered_images = self._lifting_module.render(internal_representations, batch, t=t_all).rendered_images

        self._cached_gaussian_models = internal_representations.permute(1, 0, 2)  # [B, T, D]
        if self._config.head_lvsm.use_vae:
            # Do not cast to uint8, because images are in VAE latent space (and small enough anyways, so no storage issues)
            self._denoising_history.append((x.detach().cpu() * 255))
            self._prediction_history.append((rendered_images.detach().cpu() * 255))
        else:
            self._denoising_history.append((x.detach().cpu() * 255).clamp(0, 255).to(torch.uint8))
            self._prediction_history.append((rendered_images.detach().cpu() * 255).clamp(0, 255).to(torch.uint8))

        output = DenoisingGaussianHeadLRMOutput(
            gaussian_models=internal_representations,
            diffusion_output=rendered_images,
        )

        return output

    def render(self,
               internal_representations: torch.Tensor,
               batch: GaussianHeadLRMBatch,
               t: Optional[torch.Tensor] = None,
               use_separate_renderer: bool = False) -> RenderingOutput:
        t = self._t_embedder(t)

        output = self._lifting_module.render(internal_representations, batch, t, use_separate_renderer)

        return output

    def reset_cache(self):
        self._denoising_history = []
        self._prediction_history = []
