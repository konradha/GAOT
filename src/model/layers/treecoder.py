import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from .kernels import build_balltree_with_rotation
from .ball_gemb import BallStatistics, BallGeometricEmbedding


@dataclass
class LocalGeometry:
    perm: torch.Tensor
    inverse_perm: torch.Tensor
    rot_perm: torch.Tensor
    rot_inverse_perm: torch.Tensor
    ball_stats: BallStatistics
    n_original: int
    n_padded: int
    ball_size: int

    @staticmethod
    def build(
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        ball_size: int,
        rotation_angle: float = 45.0,
    ) -> "LocalGeometry":
        perm, inverse_perm, rot_perm, rot_inverse_perm = build_balltree_with_rotation(
            pos, batch_idx, rotation_angle
        )
        ball_stats = BallStatistics.compute(pos, perm, ball_size)

        return LocalGeometry(
            perm=perm,
            inverse_perm=inverse_perm,
            rot_perm=rot_perm,
            rot_inverse_perm=rot_inverse_perm,
            ball_stats=ball_stats,
            n_original=ball_stats.n_original,
            n_padded=ball_stats.n_padded,
            ball_size=ball_size,
        )


@dataclass
class HierarchicalGeometry:
    perm: torch.Tensor
    inverse_perm: torch.Tensor
    rot_perm: torch.Tensor
    rot_inverse_perm: torch.Tensor

    level_stats: List[BallStatistics]
    level_pos: List[torch.Tensor]
    level_n: List[int]
    level_perms: List[torch.Tensor]
    level_inverse_perms: List[torch.Tensor]
    level_rot_perms: List[torch.Tensor]
    level_rot_inverse_perms: List[torch.Tensor]

    ball_sizes: List[int]
    strides: List[int]
    n_original: int
    coord_dim: int
    rotation_angle: float

    @staticmethod
    def build(
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        ball_sizes: List[int],
        strides: List[int],
        rotation_angle: float = 45.0,
    ) -> "HierarchicalGeometry":
        n_original = pos.shape[0]
        coord_dim = pos.shape[1]
        device = pos.device

        perm, inverse_perm, rot_perm, rot_inverse_perm = build_balltree_with_rotation(
            pos, batch_idx, rotation_angle
        )

        level_stats = []
        level_pos = [pos]
        level_n = [n_original]
        level_perms = [perm]
        level_inverse_perms = [inverse_perm]
        level_rot_perms = [rot_perm]
        level_rot_inverse_perms = [rot_inverse_perm]

        current_pos = pos
        current_perm = perm
        current_inv_perm = inverse_perm
        current_n = n_original

        for level_idx, ball_size in enumerate(ball_sizes):
            stats = BallStatistics.compute(current_pos, current_perm, ball_size)
            level_stats.append(stats)

            if level_idx < len(strides):
                stride = strides[level_idx]
                n_balls = stats.n_balls

                pooled_pos = stats.centroids
                n_pooled = n_balls

                if stride > 1:
                    n_pooled = (n_balls + stride - 1) // stride
                    indices = torch.arange(0, min(n_pooled * stride, n_balls), stride, device=device)
                    pooled_pos = stats.centroids[indices]
                    n_pooled = pooled_pos.shape[0]

                current_pos = pooled_pos
                current_n = n_pooled

                pooled_batch_idx = torch.zeros(current_n, dtype=torch.long, device=device)
                p, ip, rp, rip = build_balltree_with_rotation(
                    current_pos, pooled_batch_idx, rotation_angle
                )
                current_perm = p
                current_inv_perm = ip

                level_pos.append(current_pos)
                level_n.append(current_n)
                level_perms.append(p)
                level_inverse_perms.append(ip)
                level_rot_perms.append(rp)
                level_rot_inverse_perms.append(rip)

        return HierarchicalGeometry(
            perm=perm,
            inverse_perm=inverse_perm,
            rot_perm=rot_perm,
            rot_inverse_perm=rot_inverse_perm,
            level_stats=level_stats,
            level_pos=level_pos,
            level_n=level_n,
            level_perms=level_perms,
            level_inverse_perms=level_inverse_perms,
            level_rot_perms=level_rot_perms,
            level_rot_inverse_perms=level_rot_inverse_perms,
            ball_sizes=ball_sizes,
            strides=strides,
            n_original=n_original,
            coord_dim=coord_dim,
            rotation_angle=rotation_angle,
        )


class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(in_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class BallAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        coord_dim: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ball_size = ball_size
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.pe_proj = nn.Linear(coord_dim, dim)
        self.sigma = nn.Parameter(-1.0 + 0.01 * torch.randn(1, num_heads, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        rel_pos: torch.Tensor,
        pos_balls: torch.Tensor,
    ) -> torch.Tensor:
        n_padded = x.shape[0]
        ball_size = self.ball_size
        n_balls = n_padded // ball_size

        x = x + self.pe_proj(rel_pos)

        x_balls = x.view(n_balls, ball_size, self.dim)

        qkv = self.qkv(x_balls)
        qkv = qkv.view(n_balls, ball_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        pos_b = pos_balls.view(n_balls, ball_size, -1)
        dist_sq = torch.cdist(pos_b, pos_b, p=2).pow(2)
        attn_bias = self.sigma * dist_sq.unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.permute(0, 2, 1, 3).reshape(n_balls, ball_size, self.dim)
        out = out.view(n_padded, self.dim)

        out = self.proj(out)
        out = self.dropout(out)

        return out


class BallBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        coord_dim: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ball_size = ball_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BallAttention(dim, num_heads, ball_size, coord_dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        rel_pos: torch.Tensor,
        pos_balls: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rel_pos, pos_balls)
        x = x + self.mlp(self.norm2(x))
        return x


class TimeConditionedBallBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ball_size: int,
        coord_dim: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ball_size = ball_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BallAttention(dim, num_heads, ball_size, coord_dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))

        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 4),
        )

    def forward(
        self,
        x: torch.Tensor,
        rel_pos: torch.Tensor,
        pos_balls: torch.Tensor,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tau is not None:
            time_emb = self.time_mlp(tau)
            scale1, shift1, scale2, shift2 = time_emb.chunk(4, dim=-1)

            h = self.norm1(x)
            h = h * (1 + scale1) + shift1
            x = x + self.attn(h, rel_pos, pos_balls)

            h = self.norm2(x)
            h = h * (1 + scale2) + shift2
            x = x + self.mlp(h)
        else:
            x = x + self.attn(self.norm1(x), rel_pos, pos_balls)
            x = x + self.mlp(self.norm2(x))

        return x


class BallPool(nn.Module):
    def __init__(self, dim: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        stats: BallStatistics,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_padded = stats.n_padded
        ball_size = stats.ball_size
        n_balls = stats.n_balls
        dim = x.shape[-1]

        x = self.norm(x)

        if x.shape[0] < n_padded:
            pad = x[-1:].expand(n_padded - x.shape[0], -1)
            x = torch.cat([x, pad], dim=0)

        x_balls = x.view(n_balls, ball_size, dim)
        pooled = x_balls.mean(dim=1)

        if self.stride > 1:
            n_out = (n_balls + self.stride - 1) // self.stride
            indices = torch.arange(0, min(n_out * self.stride, n_balls), self.stride, device=x.device)
            pooled = pooled[indices]

        pooled_pos = stats.centroids
        if self.stride > 1:
            pooled_pos = pooled_pos[indices]

        return pooled, pooled_pos


class BallUnpool(nn.Module):
    def __init__(self, in_dim: int, skip_dim: int, out_dim: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_dim + skip_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        target_n: int,
    ) -> torch.Tensor:
        if self.stride > 1:
            expanded = x.repeat_interleave(self.stride, dim=0)
        else:
            expanded = x

        if expanded.shape[0] < target_n:
            pad_size = target_n - expanded.shape[0]
            expanded = torch.cat([expanded, expanded[-1:].expand(pad_size, -1)], dim=0)
        elif expanded.shape[0] > target_n:
            expanded = expanded[:target_n]

        combined = torch.cat([expanded, skip], dim=-1)
        out = self.proj(combined)
        out = self.norm(out)

        return out


class TreeEncoderLevel(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        ball_size: int,
        coord_dim: int = 2,
        time_conditioned: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ball_size = ball_size
        self.time_conditioned = time_conditioned

        BlockClass = TimeConditionedBallBlock if time_conditioned else BallBlock
        self.blocks = nn.ModuleList([
            BlockClass(dim, num_heads, ball_size, coord_dim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        stats: BallStatistics,
        rot_perm: Optional[torch.Tensor] = None,
        rot_inverse_perm: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_original = stats.n_original
        n_padded = stats.n_padded

        x_work = x
        if x.shape[0] < n_padded:
            x_work = torch.cat([x_work, x_work[-1:].expand(n_padded - x.shape[0], -1)], dim=0)

        pos_work = pos
        if pos.shape[0] < n_padded:
            pos_work = torch.cat([pos_work, pos_work[-1:].expand(n_padded - pos.shape[0], -1)], dim=0)

        tau_work = None
        if tau is not None:
            tau_work = tau
            if tau.shape[0] < n_padded:
                tau_work = torch.cat([tau_work, tau_work[-1:].expand(n_padded - tau.shape[0], -1)], dim=0)

        rel_pos = stats.rel_pos

        for i, block in enumerate(self.blocks):
            use_rot = rot_perm is not None and i % 2 == 1 and rot_perm.shape[0] >= n_original

            if use_rot:
                x_valid = x_work[:n_original]
                x_rot = x_valid[rot_perm[:n_original]]
                if n_padded > n_original:
                    x_rot = torch.cat([x_rot, x_rot[-1:].expand(n_padded - n_original, -1)], dim=0)

                pos_valid = pos_work[:n_original]
                pos_rot = pos_valid[rot_perm[:n_original]]
                if n_padded > n_original:
                    pos_rot = torch.cat([pos_rot, pos_rot[-1:].expand(n_padded - n_original, -1)], dim=0)

                rot_stats = BallStatistics.compute(pos[:n_original], rot_perm[:n_original], self.ball_size)
                rel_pos_rot = rot_stats.rel_pos

                if self.time_conditioned and tau_work is not None:
                    tau_valid = tau_work[:n_original]
                    tau_rot = tau_valid[rot_perm[:n_original]]
                    if n_padded > n_original:
                        tau_rot = torch.cat([tau_rot, tau_rot[-1:].expand(n_padded - n_original, -1)], dim=0)
                    x_rot = block(x_rot, rel_pos_rot, pos_rot, tau_rot)
                else:
                    x_rot = block(x_rot, rel_pos_rot, pos_rot)

                x_out = x_rot[:n_original][rot_inverse_perm[:n_original]]
                if n_padded > n_original:
                    x_work = torch.cat([x_out, x_out[-1:].expand(n_padded - n_original, -1)], dim=0)
                else:
                    x_work = x_out
            else:
                if self.time_conditioned and tau_work is not None:
                    x_work = block(x_work, rel_pos, pos_work, tau_work)
                else:
                    x_work = block(x_work, rel_pos, pos_work)

        return x_work[:n_original]


class TreeDecoderLevel(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        ball_size: int,
        coord_dim: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ball_size = ball_size

        self.blocks = nn.ModuleList([
            BallBlock(dim, num_heads, ball_size, coord_dim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        stats: BallStatistics,
        rot_perm: Optional[torch.Tensor] = None,
        rot_inverse_perm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_original = stats.n_original
        n_padded = stats.n_padded

        x_work = x
        if x.shape[0] < n_padded:
            x_work = torch.cat([x_work, x_work[-1:].expand(n_padded - x.shape[0], -1)], dim=0)

        pos_work = pos
        if pos.shape[0] < n_padded:
            pos_work = torch.cat([pos_work, pos_work[-1:].expand(n_padded - pos.shape[0], -1)], dim=0)

        rel_pos = stats.rel_pos

        for i, block in enumerate(self.blocks):
            use_rot = rot_perm is not None and i % 2 == 1 and rot_perm.shape[0] >= n_original

            if use_rot:
                x_valid = x_work[:n_original]
                x_rot = x_valid[rot_perm[:n_original]]
                if n_padded > n_original:
                    x_rot = torch.cat([x_rot, x_rot[-1:].expand(n_padded - n_original, -1)], dim=0)

                pos_valid = pos_work[:n_original]
                pos_rot = pos_valid[rot_perm[:n_original]]
                if n_padded > n_original:
                    pos_rot = torch.cat([pos_rot, pos_rot[-1:].expand(n_padded - n_original, -1)], dim=0)

                rot_stats = BallStatistics.compute(pos[:n_original], rot_perm[:n_original], self.ball_size)
                rel_pos_rot = rot_stats.rel_pos

                x_rot = block(x_rot, rel_pos_rot, pos_rot)

                x_out = x_rot[:n_original][rot_inverse_perm[:n_original]]
                if n_padded > n_original:
                    x_work = torch.cat([x_out, x_out[-1:].expand(n_padded - n_original, -1)], dim=0)
                else:
                    x_work = x_out
            else:
                x_work = block(x_work, rel_pos, pos_work)

        return x_work[:n_original]


class HierarchicalEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dims: List[int],
        ball_sizes: List[int],
        strides: List[int],
        enc_num_heads: List[int],
        enc_depths: List[int],
        latent_grid_size: Tuple[int, int],
        latent_dim: int,
        coord_dim: int = 2,
        time_conditioned: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_levels = len(ball_sizes)
        self.ball_sizes = ball_sizes
        self.strides = strides
        self.time_conditioned = time_conditioned
        self.hidden_dims = hidden_dims
        self.latent_grid_size = latent_grid_size
        self.latent_tokens = latent_grid_size[0] * latent_grid_size[1]
        self.latent_dim = latent_dim

        self.geo_embeds = nn.ModuleList([
            BallGeometricEmbedding(coord_dim, 64, hidden_dims[i])
            for i in range(self.num_levels)
        ])

        self.input_proj = nn.Linear(in_channels, hidden_dims[0])

        self.levels = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dim_projs = nn.ModuleList()

        for i in range(self.num_levels):
            self.levels.append(
                TreeEncoderLevel(
                    dim=hidden_dims[i],
                    num_heads=enc_num_heads[i],
                    depth=enc_depths[i],
                    ball_size=ball_sizes[i],
                    coord_dim=coord_dim,
                    time_conditioned=time_conditioned,
                    dropout=dropout,
                )
            )

            if i < self.num_levels - 1:
                self.pools.append(BallPool(hidden_dims[i], strides[i]))
                if hidden_dims[i] != hidden_dims[i + 1]:
                    self.dim_projs.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                else:
                    self.dim_projs.append(nn.Identity())

        self.to_latent = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        hier_geom: HierarchicalGeometry,
        tau: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        x = self.input_proj(x)

        skip_features = []
        skip_positions = []

        current_x = x
        current_pos = pos
        current_tau = tau

        for i in range(self.num_levels):
            stats = hier_geom.level_stats[i]
            rot_perm = hier_geom.level_rot_perms[i]
            rot_inv = hier_geom.level_rot_inverse_perms[i]
            inv_perm = hier_geom.level_inverse_perms[i]

            geo_feat = self.geo_embeds[i](stats, inv_perm)
            current_x = current_x + geo_feat

            current_x = self.levels[i](
                current_x,
                current_pos,
                stats,
                rot_perm=rot_perm,
                rot_inverse_perm=rot_inv,
                tau=current_tau,
            )

            skip_features.append(current_x)
            skip_positions.append(current_pos)

            if i < self.num_levels - 1:
                current_x, current_pos = self.pools[i](current_x, stats)
                current_x = self.dim_projs[i](current_x)

                if current_tau is not None:
                    n_pooled = current_x.shape[0]
                    if n_pooled <= current_tau.shape[0]:
                        current_tau = current_tau[:n_pooled]
                    else:
                        current_tau = current_tau[-1:].expand(n_pooled, -1)

        latent = self.to_latent(current_x)

        n_latent = latent.shape[0]
        if n_latent < self.latent_tokens:
            pad_size = self.latent_tokens - n_latent
            latent = torch.cat([latent, latent[-1:].expand(pad_size, -1)], dim=0)
        elif n_latent > self.latent_tokens:
            latent = latent[:self.latent_tokens]

        H, W = self.latent_grid_size
        latent = latent.view(1, H * W, self.latent_dim)

        return latent, skip_features, skip_positions


class HierarchicalDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_dims: List[int],
        ball_sizes: List[int],
        strides: List[int],
        dec_num_heads: List[int],
        dec_depths: List[int],
        latent_grid_size: Tuple[int, int],
        latent_dim: int,
        coord_dim: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_dec_levels = len(dec_depths)
        self.hidden_dims = hidden_dims
        self.ball_sizes = ball_sizes
        self.strides = strides
        self.latent_grid_size = latent_grid_size
        self.latent_tokens = latent_grid_size[0] * latent_grid_size[1]
        self.latent_dim = latent_dim

        self.from_latent = nn.Linear(latent_dim, hidden_dims[-1])

        rev_hidden_dims = hidden_dims[::-1]
        rev_ball_sizes = ball_sizes[::-1]
        rev_strides = strides[::-1]

        self.unpools = nn.ModuleList()
        self.levels = nn.ModuleList()
        self.geo_embeds = nn.ModuleList()

        for i in range(self.num_dec_levels):
            in_dim = rev_hidden_dims[i]
            skip_dim = rev_hidden_dims[i + 1] if i + 1 < len(rev_hidden_dims) else rev_hidden_dims[-1]
            out_dim = skip_dim

            stride = rev_strides[i] if i < len(rev_strides) else 1
            self.unpools.append(BallUnpool(in_dim, skip_dim, out_dim, stride))

            ball_size = rev_ball_sizes[i + 1] if i + 1 < len(rev_ball_sizes) else rev_ball_sizes[-1]
            self.levels.append(
                TreeDecoderLevel(
                    dim=out_dim,
                    num_heads=dec_num_heads[i],
                    depth=dec_depths[i],
                    ball_size=ball_size,
                    coord_dim=coord_dim,
                    dropout=dropout,
                )
            )

            self.geo_embeds.append(BallGeometricEmbedding(coord_dim, 64, out_dim))

        final_dim = rev_hidden_dims[-1] if rev_hidden_dims else hidden_dims[0]
        self.out_proj = nn.Linear(final_dim, out_channels)

    def forward(
        self,
        latent: torch.Tensor,
        skip_features: List[torch.Tensor],
        skip_positions: List[torch.Tensor],
        hier_geom: HierarchicalGeometry,
    ) -> torch.Tensor:
        latent = latent.squeeze(0)
        current_x = self.from_latent(latent)

        n_bottleneck = skip_features[-1].shape[0]
        if current_x.shape[0] > n_bottleneck:
            current_x = current_x[:n_bottleneck]
        elif current_x.shape[0] < n_bottleneck:
            pad = current_x[-1:].expand(n_bottleneck - current_x.shape[0], -1)
            current_x = torch.cat([current_x, pad], dim=0)

        rev_skip_features = skip_features[::-1]
        rev_skip_positions = skip_positions[::-1]
        rev_stats = hier_geom.level_stats[::-1]
        rev_rot_perms = hier_geom.level_rot_perms[::-1]
        rev_rot_inv_perms = hier_geom.level_rot_inverse_perms[::-1]
        rev_inv_perms = hier_geom.level_inverse_perms[::-1]

        for i in range(self.num_dec_levels):
            skip_idx = i + 1
            if skip_idx < len(rev_skip_features):
                skip = rev_skip_features[skip_idx]
                skip_pos = rev_skip_positions[skip_idx]
                target_n = skip.shape[0]

                current_x = self.unpools[i](current_x, skip, target_n)

                stats = rev_stats[skip_idx]
                rot_perm = rev_rot_perms[skip_idx]
                rot_inv = rev_rot_inv_perms[skip_idx]
                inv_perm = rev_inv_perms[skip_idx]

                geo_feat = self.geo_embeds[i](stats, inv_perm)
                current_x = current_x + geo_feat

                current_x = self.levels[i](
                    current_x,
                    skip_pos,
                    stats,
                    rot_perm=rot_perm,
                    rot_inverse_perm=rot_inv,
                )

        return self.out_proj(current_x)
