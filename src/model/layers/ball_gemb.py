import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class BallStatistics:
    centroids: torch.Tensor
    rel_pos: torch.Tensor
    distances: torch.Tensor
    dist_mean: torch.Tensor
    dist_var: torch.Tensor
    pca_eigenvalues: torch.Tensor
    ball_size: int
    n_balls: int
    n_original: int
    n_padded: int

    @staticmethod
    def compute(
        pos: torch.Tensor,
        perm: torch.Tensor,
        ball_size: int,
    ) -> "BallStatistics":
        n = pos.shape[0]
        coord_dim = pos.shape[1]
        device = pos.device
        dtype = pos.dtype

        n_padded = ((n + ball_size - 1) // ball_size) * ball_size
        n_balls = n_padded // ball_size
        pad_size = n_padded - n

        pos_perm = pos[perm]
        if pad_size > 0:
            pad_vals = pos_perm[-1:].expand(pad_size, -1)
            pos_perm = torch.cat([pos_perm, pad_vals], dim=0)

        pos_balls = pos_perm.view(n_balls, ball_size, coord_dim)

        centroids = pos_balls.mean(dim=1)
        rel_pos = pos_balls - centroids.unsqueeze(1)
        distances = rel_pos.norm(dim=-1)

        dist_mean = distances.mean(dim=1)
        dist_var = distances.var(dim=1, unbiased=False)

        cov = torch.einsum("bmd,bme->bde", rel_pos, rel_pos) / ball_size
        pca_eigenvalues = torch.linalg.eigvalsh(cov)

        return BallStatistics(
            centroids=centroids,
            rel_pos=rel_pos.reshape(n_padded, coord_dim),
            distances=distances.reshape(n_padded),
            dist_mean=dist_mean,
            dist_var=dist_var,
            pca_eigenvalues=pca_eigenvalues,
            ball_size=ball_size,
            n_balls=n_balls,
            n_original=n,
            n_padded=n_padded,
        )


@dataclass
class HierarchicalBallStatistics:
    level_stats: List[BallStatistics]
    level_centroids: List[torch.Tensor]
    n_original: int
    coord_dim: int

    @staticmethod
    def compute(
        pos: torch.Tensor,
        perm: torch.Tensor,
        ball_sizes: List[int],
        strides: List[int],
    ) -> "HierarchicalBallStatistics":
        n_original = pos.shape[0]
        coord_dim = pos.shape[1]
        device = pos.device

        level_stats = []
        level_centroids = []

        current_pos = pos
        current_perm = perm
        current_n = n_original

        for level_idx, ball_size in enumerate(ball_sizes):
            stats = BallStatistics.compute(current_pos, current_perm, ball_size)
            level_stats.append(stats)
            level_centroids.append(stats.centroids.clone())

            if level_idx < len(strides):
                stride = strides[level_idx]
                n_balls = stats.n_balls
                n_pooled = (n_balls + stride - 1) // stride

                current_pos = stats.centroids[:n_pooled * stride:stride] if n_pooled * stride <= n_balls else stats.centroids[::stride]
                actual_pooled = current_pos.shape[0]
                current_n = actual_pooled
                current_perm = torch.arange(current_n, device=device)

        return HierarchicalBallStatistics(
            level_stats=level_stats,
            level_centroids=level_centroids,
            n_original=n_original,
            coord_dim=coord_dim,
        )


class BallGeometricEmbedding(nn.Module):
    def __init__(self, coord_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.coord_dim = coord_dim
        self.out_dim = out_dim

        n_point_features = coord_dim + 1
        n_ball_features = 2 + coord_dim
        n_total = n_point_features + n_ball_features

        self.mlp = nn.Sequential(
            nn.Linear(n_total, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        stats: BallStatistics,
        inverse_perm: torch.Tensor,
    ) -> torch.Tensor:
        ball_size = stats.ball_size
        n_balls = stats.n_balls
        n_padded = stats.n_padded
        n_original = stats.n_original
        device = stats.rel_pos.device

        rel_pos = stats.rel_pos
        distances = stats.distances.unsqueeze(-1)

        dist_mean = (
            stats.dist_mean.unsqueeze(1).expand(-1, ball_size).reshape(n_padded, 1)
        )
        dist_var = (
            stats.dist_var.unsqueeze(1).expand(-1, ball_size).reshape(n_padded, 1)
        )
        pca = (
            stats.pca_eigenvalues.unsqueeze(1)
            .expand(-1, ball_size, -1)
            .reshape(n_padded, -1)
        )

        features = torch.cat([rel_pos, distances, dist_mean, dist_var, pca], dim=-1)

        feat_mean = features.mean(dim=0, keepdim=True)
        feat_std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        features = (features - feat_mean) / feat_std

        geo_embed = self.mlp(features)
        geo_embed = geo_embed[:n_original]

        output = torch.empty_like(geo_embed)
        output[inverse_perm] = geo_embed

        return output


class HierarchicalBallGeometricEmbedding(nn.Module):
    def __init__(self, coord_dim: int, hidden_dim: int, out_dims: List[int]):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_levels = len(out_dims)

        self.embeddings = nn.ModuleList([
            BallGeometricEmbedding(coord_dim, hidden_dim, out_dim)
            for out_dim in out_dims
        ])

    def forward(
        self,
        hier_stats: HierarchicalBallStatistics,
        inverse_perms: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        outputs = []
        for level_idx, (stats, inv_perm, embed) in enumerate(
            zip(hier_stats.level_stats, inverse_perms, self.embeddings)
        ):
            out = embed(stats, inv_perm)
            outputs.append(out)
        return outputs
