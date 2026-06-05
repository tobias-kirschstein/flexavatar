from dataclasses import dataclass
from enum import auto
from typing import Optional

from elias.config import Config, StringEnum


class HeadTransformerType(StringEnum):
    MESH_TOKENS = auto()
    UV_TEXTURE = auto()


class CrossAttentionType(StringEnum):
    Q2K = auto()
    Q2QK = auto()
    QK2QK = auto()


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


@dataclass
class HeadTransformerConfig(Config):
    transformer: TransformerConfig
    res_head_tokens: int
    head_transformer_type: HeadTransformerType = HeadTransformerType.MESH_TOKENS
    cross_attention_type: CrossAttentionType = CrossAttentionType.Q2K
    use_lam_transformer: bool = False
    use_lam_point_embedder: bool = False
    res_image_tokens: Optional[int] = None
    n_input_views: int = 1
    block_size_estimate_version: int = 1
    d_expression_codes: Optional[int] = None
    n_expression_tokens: Optional[int] = 4
    use_head_tokens: bool = True
    n_layers_expression_transformer: int = 4
    use_dataset_ids: bool = False
    head_template: str = 'gghead_template'


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
    use_norm_before_mlp: bool = True
    initialize_with_image: bool = False
    n_channels_color: int = 3
    fix_mlp_order: bool = False
    use_head_tokens: bool = True
    oversampling_factor: int = 1  # By how much generated (uv)-textures should be oversampled to spawn Gaussians
    use_lam_gs_decoder: bool = False
    sh_degree: int = 0
    head_template: str = 'gghead_template'
    d_expression_codes: Optional[int] = None


@dataclass
class FlexAvatarModelConfig(Config):
    head_transformer: HeadTransformerConfig
    gaussian_decoder: GaussianDecoderConfig
    patch_size: int
    in_channels: int
    n_layers_encoder: int
    n_input_views: int = 1
    use_feature_projection: bool = False
    feature_dim: int = 1536
    use_bfloat16: bool = False
    use_plucker: bool = False

    compile: bool = False
    use_gsplat: bool = False
