"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from elias.config import Config
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        if config.use_cross_attention:
            self.c_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.k_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        self.config = config

    def forward(self, x, keys: Optional[torch.Tensor] = None, poses: Optional[torch.Tensor] = None, intrinsics: Optional[torch.Tensor] = None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        if self.config.use_cross_attention:
            q = self.c_attn(x)
            k, v = self.k_attn(keys).split(self.n_embd, dim=2)
            T_key = k.shape[1]
        else:
            assert keys is None, "cross attention is disabled, but keys were given"
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            T_key = T

        k = k.view(B, T_key, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T_key, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.config.use_causal_attention)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

def get_attention_weights(q: torch.Tensor, k: torch.Tensor, gain: float = 1):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('turbo')

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)  # [B, nh, Tq, Tk]

    colored_attention_weights = cmap(att.detach().cpu().numpy() * gain)[..., :3]
    return colored_attention_weights



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

def build_repa_mlp(hidden_size: int, projector_dim: int, z_dim: int):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.config = config

    def forward(self, x: torch.Tensor,
                keys: Optional[torch.Tensor] = None,
                poses: Optional[torch.Tensor] = None,
                intrinsics: Optional[torch.Tensor] = None):
        x = x + self.attn(self.ln_1(x), keys=keys, poses=poses, intrinsics=intrinsics)
        x = x + self.mlp(self.ln_2(x))

        return x

@dataclass
class GPTConfig(Config):
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_cross_attention: bool = False
    use_repa: bool = False
    repa_layer: int = -1  # -1 is last layer
    d_repa_target: int = 768  # Target dimension for REPA, e.g., dimension of DinoV2
    d_repa_hidden: int = 2048
    use_positional_embedding: bool = True
    n_registers: int = 0  # Number of register tokens for storing global information
    use_post_layer_norm: bool = True
    n_merged_views: int = 1 # Relevant for adaptive layer norm s.t. it can be applied for each input view separately
    use_causal_attention: bool = True

    patch_size: Optional[int] = None


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.block_size is not None or not config.use_positional_embedding
        self.config = config

        transformer_dict = dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        )

        if config.use_post_layer_norm:
            transformer_dict['ln_f'] = LayerNorm(config.n_embd, bias=config.bias)

        if config.use_positional_embedding:
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)

        if config.n_registers > 0:
            transformer_dict['registers'] = nn.Embedding(config.n_registers, config.n_embd)

        self.transformer = nn.ModuleDict(transformer_dict)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        if config.use_repa:
            self._repa_mlp = build_repa_mlp(config.n_embd, config.d_repa_hidden, config.d_repa_target)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.use_positional_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor,
                keys: Optional[torch.Tensor] = None,
                poses: Optional[torch.Tensor] = None,
                intrinsics: Optional[torch.Tensor] = None
                ):
        device = x.device
        T, B, C = x.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        if self.config.use_positional_embedding:
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = x + pos_emb[:, None].to(x.dtype)  # Broadcast pos_emb across batch dimension

        x = x.permute(1, 0, 2)  # [B, T, C]
        if keys is not None:
            keys = keys.permute(1, 0, 2)

        if self.config.n_registers > 0:
            registers = self.transformer.registers.weight  # [R, C]
            registers = registers.unsqueeze(0).repeat((B, 1, 1))  # [B, R, C]
            x = torch.concat([x, registers], dim=1)

        for i, block in enumerate(self.transformer.h):
            x = block(x, keys=keys, poses=poses, intrinsics=intrinsics)

            if self.config.use_repa and ((i+1) == self.config.repa_layer or self.config.repa_layer == -1 and (i+1) == len(self.transformer.h)):
                x_repa = self._repa_mlp(x[:, :T].reshape(-1, C)).reshape(B, T, -1)

        if self.config.use_post_layer_norm:
            x = self.transformer.ln_f(x)
        x = x[:, :T]  # Discard register tokens, if any

        x = x.permute(1, 0, 2)  # [T, B, C]

        if self.config.use_repa:
            return x, x_repa
        else:
            return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu