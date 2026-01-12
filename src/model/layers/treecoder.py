import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import hashlib

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


class LocalGeometryCache:
    def __init__(self, max_size: int = 256):
        self.cache: Dict[str, LocalGeometry] = {}
        self.keys_order: List[str] = []
        self.max_size = max_size

    def _hash(self, pos: torch.Tensor, batch_idx: torch.Tensor) -> str:
        n = pos.shape[0]
        samples = [0, n // 3, 2 * n // 3, n - 1]
        data = pos[samples].cpu().numpy().tobytes()
        batch = batch_idx[samples].cpu().numpy().tobytes()
        shape = f"{pos.shape}".encode()
        return hashlib.md5(data + batch + shape).hexdigest()

    def get(
        self,
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        ball_size: int,
        rotation_angle: float,
    ) -> LocalGeometry:
        key = self._hash(pos, batch_idx)

        if key in self.cache:
            self.keys_order.remove(key)
            self.keys_order.append(key)
            return self.cache[key]

        geom = LocalGeometry.build(pos, batch_idx, ball_size, rotation_angle)

        if len(self.cache) >= self.max_size:
            old_key = self.keys_order.pop(0)
            del self.cache[old_key]

        self.cache[key] = geom
        self.keys_order.append(key)

        return geom

    def clear(self):
        self.cache.clear()
        self.keys_order.clear()


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
        tau: torch.Tensor,
    ) -> torch.Tensor:
        time_emb = self.time_mlp(tau)
        scale1, shift1, scale2, shift2 = time_emb.chunk(4, dim=-1)

        h = self.norm1(x)
        h = h * (1 + scale1) + shift1
        x = x + self.attn(h, rel_pos, pos_balls)

        h = self.norm2(x)
        h = h * (1 + scale2) + shift2
        x = x + self.mlp(h)

        return x


class TreeEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        latent_dim: int,
        latent_size: Tuple[int, int],
        num_heads: int = 8,
        num_layers: int = 4,
        ball_size: int = 64,
        coord_dim: int = 2,
        time_conditioned: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.latent_size = latent_size
        self.latent_tokens = latent_size[0] * latent_size[1]
        self.ball_size = ball_size
        self.time_conditioned = time_conditioned

        self.geo_embed = BallGeometricEmbedding(coord_dim, 64, hidden_dim)
        self.input_proj = nn.Linear(in_channels + hidden_dim, hidden_dim)

        BlockClass = TimeConditionedBallBlock if time_conditioned else BallBlock
        self.blocks = nn.ModuleList(
            [
                BlockClass(hidden_dim, num_heads, ball_size, coord_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.latent_tokens_param = nn.Parameter(
            torch.randn(1, self.latent_tokens, hidden_dim) * 0.02
        )

        self.cross_attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_kv = nn.Linear(hidden_dim, hidden_dim * 2)
        self.cross_attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.num_heads = num_heads

        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        local_geom: LocalGeometry,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_original = local_geom.n_original
        n_padded = local_geom.n_padded
        device = x.device

        geo_feat = self.geo_embed(local_geom.ball_stats, local_geom.inverse_perm)
        x = self.input_proj(torch.cat([x, geo_feat], dim=-1))

        x_perm = x[local_geom.perm]
        if n_padded > n_original:
            pad = x_perm[-1:].expand(n_padded - n_original, -1)
            x_perm = torch.cat([x_perm, pad], dim=0)

        pos_perm = pos[local_geom.perm]
        if n_padded > n_original:
            pad_pos = pos_perm[-1:].expand(n_padded - n_original, -1)
            pos_perm = torch.cat([pos_perm, pad_pos], dim=0)

        rel_pos = local_geom.ball_stats.rel_pos

        if tau is not None:
            tau_perm = tau[local_geom.perm]
            if n_padded > n_original:
                tau_perm = torch.cat(
                    [tau_perm, tau_perm[-1:].expand(n_padded - n_original, -1)], dim=0
                )

        use_rotation = True
        for i, block in enumerate(self.blocks):
            if use_rotation and i % 2 == 1:
                rot_perm = local_geom.rot_perm
                rot_inv = local_geom.rot_inverse_perm

                x_rot = x_perm[rot_perm[:n_original]]
                if n_padded > n_original:
                    x_rot = torch.cat(
                        [x_rot, x_rot[-1:].expand(n_padded - n_original, -1)], dim=0
                    )

                pos_rot = pos_perm[rot_perm[:n_original]]
                if n_padded > n_original:
                    pos_rot = torch.cat(
                        [pos_rot, pos_rot[-1:].expand(n_padded - n_original, -1)], dim=0
                    )

                rot_stats = BallStatistics.compute(
                    pos[:n_original], rot_perm[:n_original], self.ball_size
                )
                rel_pos_rot = rot_stats.rel_pos

                if self.time_conditioned and tau is not None:
                    tau_rot = tau_perm[rot_perm[:n_original]]
                    if n_padded > n_original:
                        tau_rot = torch.cat(
                            [tau_rot, tau_rot[-1:].expand(n_padded - n_original, -1)],
                            dim=0,
                        )
                    x_rot = block(x_rot, rel_pos_rot, pos_rot, tau_rot)
                else:
                    x_rot = block(x_rot, rel_pos_rot, pos_rot)

                x_out = x_rot[:n_original][rot_inv[:n_original]]
                if n_padded > n_original:
                    x_perm = torch.cat(
                        [x_out, x_out[-1:].expand(n_padded - n_original, -1)], dim=0
                    )
                else:
                    x_perm = x_out
            else:
                if self.time_conditioned and tau is not None:
                    x_perm = block(x_perm, rel_pos, pos_perm, tau_perm)
                else:
                    x_perm = block(x_perm, rel_pos, pos_perm)

        x_valid = x_perm[:n_original]
        x_orig = torch.empty_like(x_valid)
        x_orig[local_geom.inverse_perm] = x_valid

        latent = self.latent_tokens_param.expand(1, -1, -1).squeeze(0)

        q = self.cross_attn_q(latent)
        kv = self.cross_attn_kv(x_orig)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(self.latent_tokens, self.num_heads, -1).transpose(0, 1)
        k = k.view(n_original, self.num_heads, -1).transpose(0, 1)
        v = v.view(n_original, self.num_heads, -1).transpose(0, 1)

        attn_out = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        ).squeeze(0)
        attn_out = attn_out.transpose(0, 1).reshape(self.latent_tokens, self.hidden_dim)

        latent = self.cross_attn_norm(latent + self.cross_attn_proj(attn_out))
        latent = self.latent_proj(latent)

        return latent.unsqueeze(0)


class TreeDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        out_channels: int,
        latent_size: Tuple[int, int],
        num_heads: int = 8,
        num_layers: int = 2,
        ball_size: int = 64,
        coord_dim: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.latent_size = latent_size
        self.latent_tokens = latent_size[0] * latent_size[1]
        self.ball_size = ball_size
        self.num_heads = num_heads

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        self.query_proj = nn.Linear(coord_dim, hidden_dim)

        self.cross_attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_kv = nn.Linear(hidden_dim, hidden_dim * 2)
        self.cross_attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList(
            [
                BallBlock(hidden_dim, num_heads, ball_size, coord_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(
        self,
        latent: torch.Tensor,
        pos: torch.Tensor,
        local_geom: LocalGeometry,
    ) -> torch.Tensor:
        n_original = local_geom.n_original
        n_padded = local_geom.n_padded

        latent = latent.squeeze(0)
        latent = self.latent_to_hidden(latent)

        query = self.query_proj(pos)

        q = self.cross_attn_q(query)
        kv = self.cross_attn_kv(latent)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(n_original, self.num_heads, -1).transpose(0, 1)
        k = k.view(self.latent_tokens, self.num_heads, -1).transpose(0, 1)
        v = v.view(self.latent_tokens, self.num_heads, -1).transpose(0, 1)

        attn_out = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        ).squeeze(0)
        attn_out = attn_out.transpose(0, 1).reshape(n_original, self.hidden_dim)

        x = self.cross_attn_norm(query + self.cross_attn_proj(attn_out))

        x_perm = x[local_geom.perm]
        if n_padded > n_original:
            pad = x_perm[-1:].expand(n_padded - n_original, -1)
            x_perm = torch.cat([x_perm, pad], dim=0)

        pos_perm = pos[local_geom.perm]
        if n_padded > n_original:
            pad_pos = pos_perm[-1:].expand(n_padded - n_original, -1)
            pos_perm = torch.cat([pos_perm, pad_pos], dim=0)

        rel_pos = local_geom.ball_stats.rel_pos

        for block in self.blocks:
            x_perm = block(x_perm, rel_pos, pos_perm)

        x_valid = x_perm[:n_original]
        x_out = torch.empty_like(x_valid)
        x_out[local_geom.inverse_perm] = x_valid

        return self.out_proj(x_out)
