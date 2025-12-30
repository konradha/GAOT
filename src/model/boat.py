from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
from einops import rearrange

from .layers.attn import Transformer
from .balltrees import build_balltree_with_rotations


@dataclass
class TreeLevel:
    x: torch.Tensor
    pos: torch.Tensor
    mask: torch.Tensor
    rot_idx: Optional[torch.Tensor] = None


def scatter_mean(src: torch.Tensor, idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, dtype=torch.long, device=src.device)
    out.index_add_(0, idx, src)
    count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    return out / count.clamp(min=1).unsqueeze(1)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.w2(x) * F.silu(self.w1(x)))


class MPNN(nn.Module):
    def __init__(self, dim: int, steps: int, coord_dim: int):
        super().__init__()
        self.steps = steps
        self.message_fns = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * dim + coord_dim, dim), nn.GELU(), nn.LayerNorm(dim))
            for _ in range(steps)
        ])
        self.update_fns = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))
            for _ in range(steps)
        ])

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_attr = pos[edge_index[0]] - pos[edge_index[1]]
        for msg_fn, upd_fn in zip(self.message_fns, self.update_fns):
            row, col = edge_index
            msg = msg_fn(torch.cat([x[row], x[col], edge_attr], dim=-1))
            agg = scatter_mean(msg, col, x.size(0))
            x = x + upd_fn(torch.cat([x, agg], dim=-1))
        return x


class BallGeometricEmbedding(nn.Module):
    def __init__(self, coord_dim: int, feature_dim: int):
        super().__init__()
        self.coord_dim = coord_dim
        stat_dim = 3 + coord_dim
        self.mlp = nn.Sequential(
            nn.Linear(stat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU()
        )
        self.recovery = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, ball_size: int) -> torch.Tensor:
        B = ball_size
        n_balls = pos.shape[0] // B
        device = pos.device

        pos_balls = rearrange(pos, '(n b) d -> n b d', b=B)
        mask_balls = rearrange(mask, '(n b) -> n b', b=B).float()

        N_i = mask_balls.sum(dim=1)
        N_i_safe = N_i.clamp(min=1)
        has_neighbors = N_i > 0

        pos_masked = pos_balls * mask_balls.unsqueeze(-1)
        centroid = pos_masked.sum(dim=1) / N_i_safe.unsqueeze(-1)

        diff = pos_balls - centroid.unsqueeze(1)
        dist = (diff * mask_balls.unsqueeze(-1)).norm(dim=-1)

        D_avg = dist.sum(dim=1) / N_i_safe
        D_var = ((dist ** 2).sum(dim=1) / N_i_safe) - D_avg ** 2
        D_var = D_var.clamp(min=0)

        Delta = torch.zeros(n_balls, self.coord_dim, device=device)

        features = torch.cat([
            N_i.unsqueeze(1),
            D_avg.unsqueeze(1),
            D_var.unsqueeze(1),
            Delta,
        ], dim=1)

        features[~has_neighbors] = 0.0

        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        features = (features - mean) / std

        geo = self.mlp(features)
        geo = geo.unsqueeze(1).expand(-1, B, -1)
        geo = rearrange(geo, 'n b d -> (n b) d') * mask.unsqueeze(-1).float()

        combined = torch.cat([x, geo], dim=-1)
        return self.recovery(combined)


class BallAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, ball_size: int, coord_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.ball_size = ball_size

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rel_pos_proj = nn.Linear(coord_dim, dim)
        self.distance_scale = nn.Parameter(-1.0 + 0.01 * torch.randn(1, n_heads, 1, 1))

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = self.ball_size

        pos_balls = rearrange(pos, '(n b) d -> n b d', b=B)
        mask_balls = rearrange(mask, '(n b) -> n b', b=B)

        valid = mask_balls.sum(dim=1, keepdim=True).clamp(min=1).float()
        centers = (pos_balls * mask_balls.unsqueeze(-1).float()).sum(dim=1, keepdim=True) / valid.unsqueeze(-1)
        rel_pos = rearrange(pos_balls - centers, 'n b d -> (n b) d')

        x = x + self.rel_pos_proj(rel_pos) * mask.unsqueeze(-1).float()

        qkv = rearrange(self.qkv(x), '(n b) (three h d) -> three n h b d', b=B, three=3, h=self.n_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = mask_balls.unsqueeze(1) & mask_balls.unsqueeze(2)
        attn_mask = attn_mask.unsqueeze(1)

        dist = torch.cdist(pos_balls, pos_balls)
        bias = self.distance_scale * dist.unsqueeze(1)
        bias = torch.where(attn_mask, bias, torch.tensor(float('-inf'), device=bias.device, dtype=bias.dtype))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = rearrange(out, 'n h b d -> (n b) (h d)')

        return self.out_proj(out) * mask.unsqueeze(-1).float()


class BallBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ball_size: int, coord_dim: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.attn = BallAttention(dim, n_heads, ball_size, coord_dim)
        self.ffn = SwiGLU(dim, dim * 4)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pos, mask)
        x = x + self.ffn(self.norm2(x)) * mask.unsqueeze(-1).float()
        return x


class Pool(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, stride: int, coord_dim: int):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(stride * (dim_in + coord_dim), dim_out)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor):
        S = self.stride

        x_g = rearrange(x, '(n s) d -> n s d', s=S)
        pos_g = rearrange(pos, '(n s) d -> n s d', s=S)
        mask_g = rearrange(mask, '(n s) -> n s', s=S).float()

        valid = mask_g.sum(dim=1, keepdim=True).clamp(min=1)
        pos_coarse = (pos_g * mask_g.unsqueeze(-1)).sum(dim=1) / valid

        rel_pos = (pos_g - pos_coarse.unsqueeze(1)) * mask_g.unsqueeze(-1)
        x_masked = x_g * mask_g.unsqueeze(-1)

        concat = torch.cat([
            rearrange(x_masked, 'n s d -> n (s d)'),
            rearrange(rel_pos, 'n s d -> n (s d)')
        ], dim=-1)

        x_coarse = self.norm(self.proj(concat))
        mask_coarse = mask_g.sum(dim=1) > 0

        return x_coarse, pos_coarse, mask_coarse


class Unpool(nn.Module):
    def __init__(self, dim_coarse: int, dim_fine: int, stride: int, coord_dim: int):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(dim_coarse + stride * coord_dim, stride * dim_fine)
        self.norm = nn.LayerNorm(dim_fine)

    def forward(self, x_coarse: torch.Tensor, pos_coarse: torch.Tensor,
                x_fine: torch.Tensor, pos_fine: torch.Tensor, mask_fine: torch.Tensor):
        S = self.stride

        pos_fine_g = rearrange(pos_fine, '(n s) d -> n s d', s=S)
        mask_fine_g = rearrange(mask_fine, '(n s) -> n s', s=S).float()

        rel_pos = (pos_fine_g - pos_coarse.unsqueeze(1)) * mask_fine_g.unsqueeze(-1)

        concat = torch.cat([x_coarse, rearrange(rel_pos, 'n s d -> n (s d)')], dim=-1)
        delta = rearrange(self.proj(concat), 'n (s d) -> (n s) d', s=S)

        return self.norm(x_fine + delta) * mask_fine.unsqueeze(-1).float()


class EncoderStage(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, n_heads: int, ball_size: int,
                 stride: int, depth: int, coord_dim: int, use_rotation: bool):
        super().__init__()
        self.use_rotation = use_rotation
        self.ball_size = ball_size
        self.rotation_pattern = [i % 2 == 1 for i in range(depth)]

        self.geo_embed = BallGeometricEmbedding(coord_dim, dim_in)
        self.blocks = nn.ModuleList([BallBlock(dim_in, n_heads, ball_size, coord_dim) for _ in range(depth)])
        self.pool = Pool(dim_in, dim_out, stride, coord_dim)

    def forward(self, level: TreeLevel) -> tuple[TreeLevel, TreeLevel]:
        x, pos, mask = level.x, level.pos, level.mask

        x = self.geo_embed(x, pos, mask, self.ball_size)

        inv = torch.argsort(level.rot_idx) if self.use_rotation and level.rot_idx is not None else None

        for use_rot, block in zip(self.rotation_pattern, self.blocks):
            if use_rot and inv is not None:
                x = block(x[level.rot_idx], pos[level.rot_idx], mask[level.rot_idx])[inv]
            else:
                x = block(x, pos, mask)

        fine = TreeLevel(x=x, pos=pos, mask=mask, rot_idx=level.rot_idx)
        x_c, pos_c, mask_c = self.pool(x, pos, mask)
        coarse = TreeLevel(x=x_c, pos=pos_c, mask=mask_c)

        return coarse, fine


class DecoderStage(nn.Module):
    def __init__(self, dim_coarse: int, dim_fine: int, n_heads: int, ball_size: int,
                 stride: int, depth: int, coord_dim: int, use_rotation: bool):
        super().__init__()
        self.use_rotation = use_rotation
        self.ball_size = ball_size
        self.rotation_pattern = [i % 2 == 1 for i in range(depth)]

        self.unpool = Unpool(dim_coarse, dim_fine, stride, coord_dim)
        self.geo_embed = BallGeometricEmbedding(coord_dim, dim_fine)
        self.blocks = nn.ModuleList([BallBlock(dim_fine, n_heads, ball_size, coord_dim) for _ in range(depth)])

    def forward(self, coarse: TreeLevel, skip: TreeLevel) -> TreeLevel:
        x = self.unpool(coarse.x, coarse.pos, skip.x, skip.pos, skip.mask)
        pos, mask = skip.pos, skip.mask

        x = self.geo_embed(x, pos, mask, self.ball_size)

        inv = torch.argsort(skip.rot_idx) if self.use_rotation and skip.rot_idx is not None else None

        for use_rot, block in zip(self.rotation_pattern, self.blocks):
            if use_rot and inv is not None:
                x = block(x[skip.rot_idx], pos[skip.rot_idx], mask[skip.rot_idx])[inv]
            else:
                x = block(x, pos, mask)

        return TreeLevel(x=x, pos=pos, mask=mask, rot_idx=skip.rot_idx)


class SpatialCrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, out_dim: int, n_heads: int, coord_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.coord_dim = coord_dim

        self.q_proj = nn.Linear(q_dim, out_dim)
        self.k_proj = nn.Linear(kv_dim, out_dim)
        self.v_proj = nn.Linear(kv_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.q_pos_proj = nn.Linear(coord_dim, out_dim)
        self.k_pos_proj = nn.Linear(coord_dim, out_dim)

        self.distance_scale = nn.Parameter(-0.1 + 0.01 * torch.randn(1, n_heads, 1, 1))

    def forward(self, q_feat: torch.Tensor, q_pos: torch.Tensor,
                kv_feat: torch.Tensor, kv_pos: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        q = self.q_proj(q_feat) + self.q_pos_proj(q_pos)
        k = self.k_proj(kv_feat) + self.k_pos_proj(kv_pos)
        v = self.v_proj(kv_feat)

        q = rearrange(q, 'n (h d) -> h n d', h=self.n_heads)
        k = rearrange(k, 'm (h d) -> h m d', h=self.n_heads)
        v = rearrange(v, 'm (h d) -> h m d', h=self.n_heads)

        dist = torch.cdist(q_pos, kv_pos)
        bias = self.distance_scale * dist.unsqueeze(0)

        if kv_mask is not None:
            mask_bias = torch.where(kv_mask, 0.0, float('-inf'))
            bias = bias + mask_bias.unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = rearrange(out, 'h n d -> n (h d)')

        return self.out_proj(out)


class LatentCrossAttention(nn.Module):
    def __init__(self, point_dim: int, latent_dim: int, n_heads: int, coord_dim: int):
        super().__init__()
        self.to_latent = SpatialCrossAttention(coord_dim, point_dim, latent_dim, n_heads, coord_dim)
        self.from_latent = SpatialCrossAttention(coord_dim, latent_dim, point_dim, n_heads, coord_dim)

    def encode(self, point_feat: torch.Tensor, point_pos: torch.Tensor,
               point_mask: torch.Tensor, latent_pos: torch.Tensor) -> torch.Tensor:
        return self.to_latent(
            q_feat=latent_pos,
            q_pos=latent_pos,
            kv_feat=point_feat,
            kv_pos=point_pos,
            kv_mask=point_mask
        )

    def decode(self, latent_feat: torch.Tensor, latent_pos: torch.Tensor,
               point_pos: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        out = self.from_latent(
            q_feat=point_pos,
            q_pos=point_pos,
            kv_feat=latent_feat,
            kv_pos=latent_pos,
            kv_mask=None
        )
        return out * point_mask.unsqueeze(-1).float()


class BOAT(nn.Module):
    def __init__(self, input_size: int, output_size: int, config):
        super().__init__()

        tc = config.args.treecoder
        tf = config.args.transformer

        self.coord_dim = tc.coord_dim
        self.ball_sizes = tc.ball_sizes
        self.strides = tc.strides
        self.rotate_angle = tc.rotate_angle
        self.use_rotation = tc.rotate_angle > 0
        self.mpnn_steps = getattr(tc, 'mpnn_steps', 0)

        C = tc.lifting_channels
        n_heads = tc.n_heads
        n_levels = len(tc.strides)

        dims = [C * (2 ** i) for i in range(n_levels + 1)]

        self.input_proj = nn.Linear(input_size, C)

        if self.mpnn_steps > 0:
            self.mpnn = MPNN(C, self.mpnn_steps, tc.coord_dim)
        else:
            self.mpnn = None

        self.encoder_stages = nn.ModuleList([
            EncoderStage(dims[i], dims[i + 1], n_heads, tc.ball_sizes[i],
                         tc.strides[i], tc.enc_depths[i], tc.coord_dim, self.use_rotation)
            for i in range(n_levels)
        ])

        self.bottleneck_geo = BallGeometricEmbedding(tc.coord_dim, dims[-1])
        self.bottleneck = nn.ModuleList([
            BallBlock(dims[-1], n_heads, tc.ball_sizes[-1], tc.coord_dim)
            for _ in range(tc.enc_depths[-1])
        ])
        self.bottleneck_rotation = [i % 2 == 1 for i in range(tc.enc_depths[-1])]

        self.decoder_stages = nn.ModuleList([
            DecoderStage(dims[i + 1], dims[i], n_heads, tc.ball_sizes[i],
                         tc.strides[i], tc.dec_depths[i], tc.coord_dim, self.use_rotation)
            for i in range(n_levels - 1, -1, -1)
        ])

        self.output_proj = nn.Linear(C, output_size)

        latent_size = config.latent_tokens_size
        P = tf.patch_size
        self.patch_size = P

        if tc.coord_dim == 2:
            self.H, self.W, self.D = latent_size[0], latent_size[1], None
            patch_vol = P * P
            ph, pw = self.H // P, self.W // P
            positions = torch.stack(torch.meshgrid(
                torch.arange(ph, dtype=torch.float32),
                torch.arange(pw, dtype=torch.float32),
                indexing='ij'
            ), dim=-1).reshape(-1, 2)
        else:
            self.H, self.W, self.D = latent_size
            patch_vol = P ** 3
            ph, pw, pd = self.H // P, self.W // P, self.D // P
            positions = torch.stack(torch.meshgrid(
                torch.arange(ph, dtype=torch.float32),
                torch.arange(pw, dtype=torch.float32),
                torch.arange(pd, dtype=torch.float32),
                indexing='ij'
            ), dim=-1).reshape(-1, 3)

        self.register_buffer('latent_positions', positions)
        self.latent_cross_attn = LatentCrossAttention(dims[-1], C * patch_vol, n_heads, tc.coord_dim)
        self.patch_proj = nn.Linear(C * patch_vol, C * patch_vol)
        self.pos_embed_type = tf.positional_embedding

        self.processor = Transformer(input_size=C * patch_vol, output_size=C * patch_vol, config=tf)

    def _process_latent(self, z: torch.Tensor, condition: Optional[float] = None) -> torch.Tensor:
        B, N, D = z.shape
        P = self.patch_size

        if self.coord_dim == 2:
            H, W = self.H, self.W
            ph, pw = H // P, W // P
            z = z.view(B, H, W, -1)
            z = z.view(B, ph, P, pw, P, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, ph * pw, -1)
        else:
            H, W, D_ = self.H, self.W, self.D
            ph, pw, pd = H // P, W // P, D_ // P
            z = z.view(B, H, W, D_, -1)
            z = z.view(B, ph, P, pw, P, pd, P, -1).permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(B, ph * pw * pd, -1)

        z = self.patch_proj(z)

        if self.pos_embed_type == 'absolute':
            pos = self.latent_positions
            dim_per = z.shape[-1] // (2 * pos.shape[-1])
            freq = torch.arange(dim_per, device=z.device, dtype=torch.float32)
            inv_freq = 1.0 / (10000 ** (freq / dim_per))
            sincos = pos[:, :, None] * inv_freq
            pe = torch.cat([sincos.sin(), sincos.cos()], dim=-1).view(pos.shape[0], -1)
            z = z + pe
            rel_pos = None
        else:
            rel_pos = self.latent_positions

        z = self.processor(z, condition=condition, relative_positions=rel_pos)

        if self.coord_dim == 2:
            z = z.view(B, ph, pw, P, P, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, -1)
        else:
            z = z.view(B, ph, pw, pd, P, P, P, -1).permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, H * W * D_, -1)

        return z

    def forward(self, latent_tokens_coord: torch.Tensor, xcoord: torch.Tensor,
                pndata: torch.Tensor, query_coord: Optional[torch.Tensor] = None,
                encoder_nbrs: Optional[list] = None, decoder_nbrs: Optional[list] = None,
                condition: Optional[float] = None) -> torch.Tensor:

        B, N, _ = pndata.shape
        device = pndata.device

        if xcoord.dim() == 2:
            xcoord = xcoord.unsqueeze(0).expand(B, -1, -1)
        if latent_tokens_coord.dim() == 2:
            latent_tokens_coord = latent_tokens_coord.unsqueeze(0).expand(B, -1, -1)

        outputs = []

        for b in range(B):
            pos = xcoord[b]
            feat = pndata[b]
            latent_pos = latent_tokens_coord[b]

            batch_idx = torch.zeros(N, dtype=torch.long, device=device)
            tree_idx, tree_mask, rot_indices = build_balltree_with_rotations(
                pos, batch_idx, self.strides, self.ball_sizes, self.rotate_angle
            )

            x = self.input_proj(feat)

            if self.mpnn is not None:
                if encoder_nbrs is None:
                    raise ValueError("encoder_nbrs required when mpnn_steps > 0")
                edge_index = encoder_nbrs[b] if isinstance(encoder_nbrs, list) else encoder_nbrs
                x = self.mpnn(x, pos, edge_index)

            x = x[tree_idx]
            pos_tree = pos[tree_idx]

            level = TreeLevel(x=x, pos=pos_tree, mask=tree_mask,
                              rot_idx=rot_indices[0] if rot_indices else None)

            skips = []
            for i, stage in enumerate(self.encoder_stages):
                level.rot_idx = rot_indices[i] if rot_indices and i < len(rot_indices) else None
                level, skip = stage(level)
                skips.append(skip)

            level.x = self.bottleneck_geo(level.x, level.pos, level.mask, self.ball_sizes[-1])

            bn_rot = rot_indices[len(self.encoder_stages)] if rot_indices and len(rot_indices) > len(self.encoder_stages) else None
            inv = torch.argsort(bn_rot) if bn_rot is not None else None

            for use_rot, block in zip(self.bottleneck_rotation, self.bottleneck):
                if use_rot and inv is not None:
                    level.x = block(level.x[bn_rot], level.pos[bn_rot], level.mask[bn_rot])[inv]
                else:
                    level.x = block(level.x, level.pos, level.mask)

            z = self.latent_cross_attn.encode(level.x, level.pos, level.mask, latent_pos)
            z = self._process_latent(z.unsqueeze(0), condition).squeeze(0)
            level.x = self.latent_cross_attn.decode(z, latent_pos, level.pos, level.mask)

            for i, stage in enumerate(self.decoder_stages):
                level = stage(level, skips[-(i + 1)])

            out = self.output_proj(level.x) * level.mask.unsqueeze(-1).float()

            result = torch.zeros(N, out.shape[-1], device=device, dtype=out.dtype)
            result[tree_idx[tree_mask]] = out[tree_mask]

            outputs.append(result)

        return torch.stack(outputs, dim=0)

    def autoregressive_predict(self, x_batch: torch.Tensor, time_indices: np.ndarray,
                               t_values: np.ndarray, stats: Dict, stepper_mode: str = "output",
                               latent_tokens_coord: Optional[torch.Tensor] = None,
                               fixed_coord: Optional[torch.Tensor] = None,
                               encoder_nbrs: Optional[List] = None,
                               decoder_nbrs: Optional[List] = None,
                               use_conditional_norm: bool = False) -> torch.Tensor:

        batch_size, num_nodes, _ = x_batch.shape
        device = x_batch.device
        num_timesteps = len(time_indices)

        u_mean = stats["u"]["mean"].to(device)
        u_std = stats["u"]["std"].to(device)
        u_dim = u_mean.shape[0]

        c_dim = stats["c"]["mean"].shape[0] if "c" in stats else 0
        c_feat = x_batch[..., u_dim:u_dim + c_dim] if c_dim > 0 else None

        current_u = x_batch[..., :u_dim]
        predictions = []

        for idx in range(1, num_timesteps):
            t_in, t_out = time_indices[idx - 1], time_indices[idx]
            dt = t_values[t_out] - t_values[t_in]

            t_start_norm = (t_values[t_in] - stats["start_time"]["mean"]) / stats["start_time"]["std"]
            dt_norm = (dt - stats["time_diffs"]["mean"]) / stats["time_diffs"]["std"]

            t_start_feat = torch.full((batch_size, num_nodes, 1), t_start_norm, device=device, dtype=x_batch.dtype)
            dt_feat = torch.full((batch_size, num_nodes, 1), dt_norm, device=device, dtype=x_batch.dtype)

            inputs = [current_u]
            if c_feat is not None:
                inputs.append(c_feat)
            inputs.extend([t_start_feat, dt_feat])
            x_in = torch.cat(inputs, dim=-1)

            with torch.no_grad():
                pred = self.forward(latent_tokens_coord, fixed_coord, x_in,
                                    encoder_nbrs=encoder_nbrs, decoder_nbrs=decoder_nbrs)

                if stepper_mode == "output":
                    pred_denorm = pred * u_std + u_mean
                elif stepper_mode == "residual":
                    res_mean = stats["res"]["mean"].to(device)
                    res_std = stats["res"]["std"].to(device)
                    pred_denorm = (current_u * u_std + u_mean) + (pred * res_std + res_mean)
                elif stepper_mode == "time_der":
                    der_mean = stats["der"]["mean"].to(device)
                    der_std = stats["der"]["std"].to(device)
                    pred_denorm = (current_u * u_std + u_mean) + dt * (pred * der_std + der_mean)
                else:
                    raise ValueError(f"Unknown stepper_mode: {stepper_mode}")

                predictions.append(pred_denorm)
                current_u = (pred_denorm - u_mean) / u_std

        return torch.stack(predictions, dim=1)
