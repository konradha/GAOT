import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

from .layers.attn import Transformer, TransformerConfig
from .layers.treecoder import (
    LocalGeometry,
    LocalGeometryCache,
    TreeEncoder,
    TreeDecoder,
)


@dataclass
class BOATConfig:
    latent_tokens_size: Tuple[int, int] = (64, 64)
    hidden_dim: int = 128
    latent_dim: int = 64
    num_heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 2
    ball_size: int = 64
    coord_dim: int = 2
    rotation_angle: float = 45.0
    time_conditioned: bool = True
    dropout: float = 0.0
    transformer: TransformerConfig = field(default_factory=TransformerConfig)


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
        self.latent_size = config.latent_tokens_size
        self.latent_tokens = self.latent_size[0] * self.latent_size[1]

        self.H, self.W = self.latent_size
        self.patch_size = config.transformer.patch_size
        self.node_latent_size = config.latent_dim

        patch_volume = self.patch_size * self.patch_size
        self.patch_dim = patch_volume * config.latent_dim

        self.encoder = TreeEncoder(
            in_channels=input_size,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            latent_size=config.latent_tokens_size,
            num_heads=config.num_heads,
            num_layers=config.encoder_layers,
            ball_size=config.ball_size,
            coord_dim=config.coord_dim,
            time_conditioned=config.time_conditioned,
            dropout=config.dropout,
        )

        self.patch_linear = nn.Linear(self.patch_dim, self.patch_dim)

        self.positional_embedding_name = config.transformer.positional_embedding
        self.positions = self._get_patch_positions()

        self.processor = Transformer(
            input_size=self.patch_dim,
            output_size=self.patch_dim,
            config=config.transformer,
        )

        self.decoder = TreeDecoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            out_channels=output_size,
            latent_size=config.latent_tokens_size,
            num_heads=config.num_heads,
            num_layers=config.decoder_layers,
            ball_size=config.ball_size,
            coord_dim=config.coord_dim,
            dropout=config.dropout,
        )

        self.geom_cache = LocalGeometryCache(max_size=256)

    def _get_patch_positions(self):
        P = self.patch_size
        num_patches_H = self.H // P
        num_patches_W = self.W // P

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
    ) -> LocalGeometry:
        if use_cache:
            return self.geom_cache.get(
                pos,
                batch_idx,
                self.config.ball_size,
                self.config.rotation_angle,
            )
        return LocalGeometry.build(
            pos,
            batch_idx,
            self.config.ball_size,
            self.config.rotation_angle,
        )

    def process(
        self,
        rndata: torch.Tensor,
        condition: Optional[float] = None,
    ) -> torch.Tensor:
        batch_size = rndata.shape[0]
        n_regional_nodes = rndata.shape[1]
        C = rndata.shape[2]
        P = self.patch_size
        H, W = self.H, self.W

        assert n_regional_nodes == H * W
        assert H % P == 0 and W % P == 0

        num_patches_H = H // P
        num_patches_W = W // P

        rndata = rndata.view(batch_size, H, W, C)
        rndata = rndata.view(batch_size, num_patches_H, P, num_patches_W, P, C)
        rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
        rndata = rndata.view(batch_size, num_patches_H * num_patches_W, P * P * C)

        rndata = self.patch_linear(rndata)
        pos = self.positions.to(rndata.device)

        if self.positional_embedding_name == "absolute":
            pos_emb = self._compute_absolute_embeddings(pos, self.patch_dim)
            rndata = rndata + pos_emb
            relative_positions = None
        elif self.positional_embedding_name == "rope":
            relative_positions = pos
        else:
            relative_positions = None

        rndata = self.processor(
            rndata, condition=condition, relative_positions=relative_positions
        )

        rndata = rndata.view(batch_size, num_patches_H, num_patches_W, P, P, C)
        rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
        rndata = rndata.view(batch_size, H * W, C)

        return rndata

    def forward(
        self,
        pos: torch.Tensor,
        features: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        local_geom: Optional[LocalGeometry] = None,
    ) -> torch.Tensor:
        n = pos.shape[0]
        device = pos.device

        if batch_idx is None:
            batch_idx = torch.zeros(n, dtype=torch.long, device=device)

        if local_geom is None:
            local_geom = self.get_geometry(pos, batch_idx)

        if tau is not None:
            if tau.dim() == 0:
                tau = tau.expand(n).unsqueeze(-1)
            elif tau.dim() == 1:
                tau = tau.unsqueeze(-1)

        latent = self.encoder(features, pos, local_geom, tau)
        latent = self.process(latent, condition=None)
        output = self.decoder(latent, pos, local_geom)

        return output

    def forward_with_geometry(
        self,
        pos: torch.Tensor,
        features: torch.Tensor,
        local_geom: LocalGeometry,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = pos.shape[0]

        if tau is not None:
            if tau.dim() == 0:
                tau = tau.expand(n).unsqueeze(-1)
            elif tau.dim() == 1:
                tau = tau.unsqueeze(-1)

        latent = self.encoder(features, pos, local_geom, tau)
        latent = self.process(latent, condition=None)
        output = self.decoder(latent, pos, local_geom)

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

        local_geom = self.get_geometry(pos, batch_idx, use_cache=True)

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

            pred_der_norm = self.forward_with_geometry(pos, features, local_geom, tau)

            if der_stats is not None and tau_val in der_stats:
                der_mean, der_std = der_stats[tau_val]
                pred_der = pred_der_norm * der_std + der_mean
            else:
                pred_der = pred_der_norm

            u_current = u_current + pred_der * tau_val
            trajectory.append(u_current)

        return trajectory
