import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass, field

from .layers.attn import Transformer, TransformerConfig
from .layers.treecoder import (
    LocalGeometry,
    HierarchicalGeometry,
    LocalGeometryCache,
    BallStatistics,
    BallBlock,
    TimeConditionedBallBlock,
    BallPool,
    BallUnpool,
    BallGeometricEmbedding,
    SwiGLU,
    HierarchicalEncoder,
    HierarchicalDecoder
)


@dataclass
class BOATConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    ball_sizes: List[int] = field(default_factory=lambda: [128, 128, 64])
    strides: List[int] = field(default_factory=lambda: [2, 2])
    enc_num_heads: List[int] = field(default_factory=lambda: [4, 8, 16])
    enc_depths: List[int] = field(default_factory=lambda: [2, 2, 4])
    dec_num_heads: List[int] = field(default_factory=lambda: [8, 4])
    dec_depths: List[int] = field(default_factory=lambda: [2, 2])

    latent_grid_size: Tuple[int, int] = (64, 64)
    latent_dim: int = 64

    coord_dim: int = 2
    rotation_angle: float = 45.0
    time_conditioned: bool = True
    dropout: float = 0.0

    transformer: TransformerConfig = field(default_factory=TransformerConfig)


class EncoderLevel(nn.Module):
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
        self.dim = dim

        BlockClass = TimeConditionedBallBlock if time_conditioned else BallBlock
        self.blocks = nn.ModuleList(
            [
                BlockClass(dim, num_heads, ball_size, coord_dim, dropout=dropout)
                for _ in range(depth)
            ]
        )

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
        device = x.device

        x_perm = x
        if x.shape[0] < n_padded:
            pad = x_perm[-1:].expand(n_padded - x.shape[0], -1)
            x_perm = torch.cat([x_perm, pad], dim=0)

        pos_perm = pos
        if pos.shape[0] < n_padded:
            pad_pos = pos_perm[-1:].expand(n_padded - pos.shape[0], -1)
            pos_perm = torch.cat([pos_perm, pad_pos], dim=0)

        rel_pos = stats.rel_pos

        tau_perm = None
        if tau is not None:
            tau_perm = tau
            if tau.shape[0] < n_padded:
                tau_perm = torch.cat(
                    [tau, tau[-1:].expand(n_padded - tau.shape[0], -1)], dim=0
                )

        for i, block in enumerate(self.blocks):
            use_rot = (
                rot_perm is not None and i % 2 == 1 and rot_perm.shape[0] >= n_original
            )

            if use_rot:
                x_valid = x_perm[:n_original]
                x_rot = x_valid[rot_perm[:n_original]]
                if n_padded > n_original:
                    x_rot = torch.cat(
                        [x_rot, x_rot[-1:].expand(n_padded - n_original, -1)], dim=0
                    )

                pos_valid = pos_perm[:n_original]
                pos_rot = pos_valid[rot_perm[:n_original]]
                if n_padded > n_original:
                    pos_rot = torch.cat(
                        [pos_rot, pos_rot[-1:].expand(n_padded - n_original, -1)], dim=0
                    )

                rot_stats = BallStatistics.compute(
                    pos[:n_original], rot_perm[:n_original], self.ball_size
                )
                rel_pos_rot = rot_stats.rel_pos

                if self.time_conditioned and tau_perm is not None:
                    tau_valid = tau_perm[:n_original]
                    tau_rot = tau_valid[rot_perm[:n_original]]
                    if n_padded > n_original:
                        tau_rot = torch.cat(
                            [tau_rot, tau_rot[-1:].expand(n_padded - n_original, -1)],
                            dim=0,
                        )
                    x_rot = block(x_rot, rel_pos_rot, pos_rot, tau_rot)
                else:
                    x_rot = block(x_rot, rel_pos_rot, pos_rot)

                x_out = x_rot[:n_original][rot_inverse_perm[:n_original]]
                if n_padded > n_original:
                    x_perm = torch.cat(
                        [x_out, x_out[-1:].expand(n_padded - n_original, -1)], dim=0
                    )
                else:
                    x_perm = x_out
            else:
                if self.time_conditioned and tau_perm is not None:
                    x_perm = block(x_perm, rel_pos, pos_perm, tau_perm)
                else:
                    x_perm = block(x_perm, rel_pos, pos_perm)

        return x_perm[:n_original]


class DecoderLevel(nn.Module):
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
        self.dim = dim

        self.blocks = nn.ModuleList(
            [
                BallBlock(dim, num_heads, ball_size, coord_dim, dropout=dropout)
                for _ in range(depth)
            ]
        )

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

        x_perm = x
        if x.shape[0] < n_padded:
            pad = x_perm[-1:].expand(n_padded - x.shape[0], -1)
            x_perm = torch.cat([x_perm, pad], dim=0)

        pos_perm = pos
        if pos.shape[0] < n_padded:
            pad_pos = pos_perm[-1:].expand(n_padded - pos.shape[0], -1)
            pos_perm = torch.cat([pos_perm, pad_pos], dim=0)

        rel_pos = stats.rel_pos

        for i, block in enumerate(self.blocks):
            use_rot = (
                rot_perm is not None and i % 2 == 1 and rot_perm.shape[0] >= n_original
            )

            if use_rot:
                x_valid = x_perm[:n_original]
                x_rot = x_valid[rot_perm[:n_original]]
                if n_padded > n_original:
                    x_rot = torch.cat(
                        [x_rot, x_rot[-1:].expand(n_padded - n_original, -1)], dim=0
                    )

                pos_valid = pos_perm[:n_original]
                pos_rot = pos_valid[rot_perm[:n_original]]
                if n_padded > n_original:
                    pos_rot = torch.cat(
                        [pos_rot, pos_rot[-1:].expand(n_padded - n_original, -1)], dim=0
                    )

                rot_stats = BallStatistics.compute(
                    pos[:n_original], rot_perm[:n_original], self.ball_size
                )
                rel_pos_rot = rot_stats.rel_pos

                x_rot = block(x_rot, rel_pos_rot, pos_rot)

                x_out = x_rot[:n_original][rot_inverse_perm[:n_original]]
                if n_padded > n_original:
                    x_perm = torch.cat(
                        [x_out, x_out[-1:].expand(n_padded - n_original, -1)], dim=0
                    )
                else:
                    x_perm = x_out
            else:
                x_perm = block(x_perm, rel_pos, pos_perm)

        return x_perm[:n_original]


class BOAT(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: Optional[BOATConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = BOATConfig()

        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.coord_dim = config.coord_dim

        H, W = config.latent_grid_size
        self.latent_tokens = H * W
        self.patch_size = config.transformer.patch_size
        patch_volume = self.patch_size * self.patch_size
        self.patch_dim = patch_volume * config.latent_dim

        self.encoder = HierarchicalEncoder(
            in_channels=input_size,
            hidden_dims=config.hidden_dims,
            ball_sizes=config.ball_sizes,
            strides=config.strides,
            enc_num_heads=config.enc_num_heads,
            enc_depths=config.enc_depths,
            latent_grid_size=config.latent_grid_size,
            latent_dim=config.latent_dim,
            coord_dim=config.coord_dim,
            time_conditioned=config.time_conditioned,
            dropout=config.dropout,
        )

        self.patch_linear = nn.Linear(self.patch_dim, self.patch_dim)
        self.positional_embedding_name = config.transformer.positional_embedding
        self.positions = self._get_patch_positions(config.latent_grid_size)

        self.processor = Transformer(
            input_size=self.patch_dim,
            output_size=self.patch_dim,
            config=config.transformer,
        )

        self.unpatch_linear = nn.Linear(self.patch_dim, self.patch_dim)

        self.decoder = HierarchicalDecoder(
            out_channels=output_size,
            hidden_dims=config.hidden_dims,
            ball_sizes=config.ball_sizes,
            strides=config.strides,
            dec_num_heads=config.dec_num_heads,
            dec_depths=config.dec_depths,
            latent_grid_size=config.latent_grid_size,
            latent_dim=config.latent_dim,
            coord_dim=config.coord_dim,
            dropout=config.dropout,
        )

        self.geom_cache = LocalGeometryCache(max_size=256)

    def _get_patch_positions(self, latent_grid_size: Tuple[int, int]):
        H, W = latent_grid_size
        P = self.patch_size
        num_patches_H = H // P
        num_patches_W = W // P

        positions = torch.stack(
            torch.meshgrid(
                torch.arange(num_patches_H, dtype=torch.float32),
                torch.arange(num_patches_W, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)

        return positions

    def _compute_absolute_embeddings(self, positions, embed_dim):
        num_pos_dims = positions.size(1)
        dim_touse = embed_dim // (2 * num_pos_dims)
        freq_seq = torch.arange(dim_touse, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (10000 ** (freq_seq / dim_touse))
        sinusoid_inp = positions[:, :, None] * inv_freq[None, None, :]
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.view(positions.size(0), -1)
        return pos_emb

    def get_geometry(
        self,
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
        use_cache: bool = True,
    ) -> HierarchicalGeometry:
        if use_cache:
            return self.geom_cache.get_hierarchical(
                pos,
                batch_idx,
                self.config.ball_sizes,
                self.config.strides,
                self.config.rotation_angle,
            )
        return HierarchicalGeometry.build(
            pos,
            batch_idx,
            self.config.ball_sizes,
            self.config.strides,
            self.config.rotation_angle,
        )

    def process_latent(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        H, W = self.config.latent_grid_size
        C = self.config.latent_dim
        P = self.patch_size

        assert H % P == 0 and W % P == 0
        num_patches_H = H // P
        num_patches_W = W // P

        latent = latent.view(batch_size, H, W, C)
        latent = latent.view(batch_size, num_patches_H, P, num_patches_W, P, C)
        latent = latent.permute(0, 1, 3, 2, 4, 5).contiguous()
        latent = latent.view(batch_size, num_patches_H * num_patches_W, P * P * C)

        latent = self.patch_linear(latent)
        pos = self.positions.to(latent.device)

        if self.positional_embedding_name == "absolute":
            pos_emb = self._compute_absolute_embeddings(pos, self.patch_dim)
            latent = latent + pos_emb
            relative_positions = None
        elif self.positional_embedding_name == "rope":
            relative_positions = pos
        else:
            relative_positions = None

        latent = self.processor(
            latent, condition=None, relative_positions=relative_positions
        )

        latent = self.unpatch_linear(latent)

        latent = latent.view(batch_size, num_patches_H, num_patches_W, P, P, C)
        latent = latent.permute(0, 1, 3, 2, 4, 5).contiguous()
        latent = latent.view(batch_size, H * W, C)

        return latent

    def forward(
        self,
        pos: torch.Tensor,
        features: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        hier_geom: Optional[HierarchicalGeometry] = None,
    ) -> torch.Tensor:
        n = pos.shape[0]
        device = pos.device

        if batch_idx is None:
            batch_idx = torch.zeros(n, dtype=torch.long, device=device)

        if hier_geom is None:
            hier_geom = self.get_geometry(pos, batch_idx)

        if tau is not None:
            if tau.dim() == 0:
                tau = tau.expand(n).unsqueeze(-1)
            elif tau.dim() == 1:
                tau = tau.unsqueeze(-1)

        latent, skip_features, skip_positions = self.encoder(
            features, pos, hier_geom, tau
        )
        latent = self.process_latent(latent)
        output = self.decoder(latent, skip_features, skip_positions, hier_geom)

        return output

    def forward_with_geometry(
        self,
        pos: torch.Tensor,
        features: torch.Tensor,
        hier_geom: HierarchicalGeometry,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = pos.shape[0]

        if tau is not None:
            if tau.dim() == 0:
                tau = tau.expand(n).unsqueeze(-1)
            elif tau.dim() == 1:
                tau = tau.unsqueeze(-1)

        latent, skip_features, skip_positions = self.encoder(
            features, pos, hier_geom, tau
        )
        latent = self.process_latent(latent)
        output = self.decoder(latent, skip_features, skip_positions, hier_geom)

        return output


class BOATForPDE(BOAT):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: Optional[BOATConfig] = None,
    ):
        if config is None:
            config = BOATConfig()
        config.time_conditioned = True
        super().__init__(input_size, output_size, config)

    def predict_derivative(
        self,
        pos: torch.Tensor,
        u: torch.Tensor,
        tau: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        u_mean: Optional[torch.Tensor] = None,
        u_std: Optional[torch.Tensor] = None,
        extra_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if u_mean is not None and u_std is not None:
            u_norm = (u - u_mean) / u_std
        else:
            u_norm = u

        if extra_features is not None:
            features = torch.cat([u_norm, extra_features], dim=-1)
        else:
            features = u_norm

        return self.forward(pos, features, batch_idx, tau)

    def rollout(
        self,
        pos: torch.Tensor,
        u_init: torch.Tensor,
        tau_schedule: List[int],
        batch_idx: Optional[torch.Tensor] = None,
        u_mean: Optional[torch.Tensor] = None,
        u_std: Optional[torch.Tensor] = None,
        der_stats: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        extra_features: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        n = pos.shape[0]
        device = pos.device

        if batch_idx is None:
            batch_idx = torch.zeros(n, dtype=torch.long, device=device)

        hier_geom = self.get_geometry(pos, batch_idx, use_cache=True)

        u_current = u_init.clone()
        trajectory = [u_init]

        for tau_val in tau_schedule:
            tau = torch.full((n,), float(tau_val), device=device, dtype=pos.dtype)

            if u_mean is not None and u_std is not None:
                u_norm = (u_current - u_mean) / u_std
            else:
                u_norm = u_current

            if extra_features is not None:
                features = torch.cat([u_norm, extra_features], dim=-1)
            else:
                features = u_norm

            pred_der_norm = self.forward_with_geometry(pos, features, hier_geom, tau)

            if der_stats is not None and tau_val in der_stats:
                der_mean, der_std = der_stats[tau_val]
                pred_der = pred_der_norm * der_std + der_mean
            else:
                pred_der = pred_der_norm

            u_current = u_current + pred_der * tau_val
            trajectory.append(u_current)

        return trajectory


def build_boat_small_config():
    return BOATConfig(
        hidden_dims=[32, 64, 128],
        ball_sizes=[256, 128, 64],
        strides=[4, 4],
        enc_num_heads=[2, 4, 8],
        enc_depths=[2, 2, 4],
        dec_num_heads=[4, 2],
        dec_depths=[2, 2],
        latent_grid_size=(64, 64),
        latent_dim=64,
        coord_dim=2,
        rotation_angle=45.0,
        time_conditioned=True,
        dropout=0.0,
        transformer=TransformerConfig(
            patch_size=2,
            hidden_size=256,
            num_layers=3,
            positional_embedding="absolute",
        ),
    )


def build_boat_medium_config():
    return BOATConfig(
        hidden_dims=[64, 128, 256],
        ball_sizes=[256, 128, 64],
        strides=[4, 4],
        enc_num_heads=[4, 8, 16],
        enc_depths=[2, 4, 6],
        dec_num_heads=[8, 4],
        dec_depths=[4, 2],
        latent_grid_size=(64, 64),
        latent_dim=64,
        coord_dim=2,
        rotation_angle=45.0,
        time_conditioned=True,
        dropout=0.0,
        transformer=TransformerConfig(
            patch_size=2,
            hidden_size=256,
            num_layers=6,
            positional_embedding="absolute",
        ),
    )


def build_boat_large_config():
    return BOATConfig(
        hidden_dims=[128, 256, 512],
        ball_sizes=[256, 128, 64],
        strides=[4, 4],
        enc_num_heads=[8, 16, 32],
        enc_depths=[2, 4, 8],
        dec_num_heads=[16, 8],
        dec_depths=[4, 2],
        latent_grid_size=(64, 64),
        latent_dim=128,
        coord_dim=2,
        rotation_angle=45.0,
        time_conditioned=True,
        dropout=0.0,
        transformer=TransformerConfig(
            patch_size=2,
            hidden_size=512,
            num_layers=8,
            positional_embedding="absolute",
        ),
    )
