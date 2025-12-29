import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from einops import rearrange, reduce

from .layers.attn import Transformer
from .balltrees import build_balltree_with_rotations


@dataclass
class TreeNode:
    x: torch.Tensor
    pos: torch.Tensor
    batch_idx: torch.Tensor
    tree_idx: torch.Tensor
    tree_mask: torch.Tensor
    tree_idx_rot: Optional[torch.Tensor] = None
    children: Optional[TreeNode] = None


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.w2(x) * F.silu(self.w1(x)))


class BallAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, ball_size: int, coord_dim: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.ball_size = ball_size
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rel_pos_proj = nn.Linear(coord_dim, dim)
        self.distance_scale = nn.Parameter(-1.0 + 0.01 * torch.randn(1, n_heads, 1, 1))

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        B = self.ball_size

        centers = reduce(pos, "(n b) d -> n d", "mean", b=B)
        pos_in_balls = rearrange(pos, "(n b) d -> n b d", b=B)
        rel_pos = pos_in_balls - centers.unsqueeze(1)
        rel_pos_flat = rearrange(rel_pos, "n b d -> (n b) d")

        x = x + self.rel_pos_proj(rel_pos_flat)

        qkv = rearrange(
            self.qkv(x),
            "(n b) (h d three) -> three n h b d",
            b=B,
            h=self.n_heads,
            three=3,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        dist = torch.cdist(pos_in_balls, pos_in_balls)
        attn_bias = self.distance_scale * dist.unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = rearrange(out, "n h b d -> (n b) (h d)")

        return self.out_proj(out)


class BallTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ball_size: int, coord_dim: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.attn = BallAttention(dim, n_heads, ball_size, coord_dim)
        self.ffn = SwiGLU(dim, dim * 4)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.ffn(self.norm2(x))
        return x


class BallPooling(nn.Module):
    def __init__(self, dim: int, stride: int, coord_dim: int):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(stride * dim + stride * coord_dim, stride * dim)
        self.norm = nn.BatchNorm1d(stride * dim)

    def forward(self, node: TreeNode) -> TreeNode:
        S = self.stride

        centers = reduce(node.pos, "(n s) d -> n d", "mean", s=S)
        pos_grouped = rearrange(node.pos, "(n s) d -> n s d", s=S)

        x = torch.cat(
            [
                rearrange(node.x, "(n s) c -> n (s c)", s=S),
                rearrange(pos_grouped - centers.unsqueeze(1), "n s d -> n (s d)"),
            ],
            dim=-1,
        )

        x = self.norm(self.proj(x))

        return TreeNode(
            x=x,
            pos=centers,
            batch_idx=node.batch_idx[::S].contiguous(),
            tree_idx=node.tree_idx[::S].contiguous(),
            tree_mask=node.tree_mask[::S].contiguous(),
            tree_idx_rot=None,
            children=node,
        )


class BallUnpooling(nn.Module):
    def __init__(self, dim: int, stride: int, coord_dim: int):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(stride * dim + stride * coord_dim, stride * dim)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, node: TreeNode) -> TreeNode:
        S = self.stride
        child = node.children

        child_pos_grouped = rearrange(child.pos, "(n s) d -> n s d", s=S)
        rel_pos = child_pos_grouped - node.pos.unsqueeze(1)

        x = torch.cat([node.x, rearrange(rel_pos, "n s d -> n (s d)")], dim=-1)

        refined = rearrange(self.proj(x), "n (s d) -> (n s) d", s=S)
        child.x = self.norm(child.x + refined)

        return child


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        ball_size: int,
        stride: int,
        depth: int,
        coord_dim: int,
        use_rotation: bool,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                BallTransformerBlock(dim, n_heads, ball_size, coord_dim)
                for _ in range(depth)
            ]
        )
        self.pool = BallPooling(dim, stride, coord_dim)

        self.use_rotation = use_rotation
        self.rotate_pattern = [i % 2 == 1 for i in range(depth)]

    def forward(self, node: TreeNode) -> TreeNode:
        if self.use_rotation and node.tree_idx_rot is not None:
            inv_perm = torch.argsort(node.tree_idx_rot)

        for rotate, block in zip(self.rotate_pattern, self.blocks):
            if rotate and self.use_rotation and node.tree_idx_rot is not None:
                x_rot = node.x[node.tree_idx_rot]
                pos_rot = node.pos[node.tree_idx_rot]
                node.x = block(x_rot, pos_rot)[inv_perm]
            else:
                node.x = block(node.x, node.pos)

        return self.pool(node)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        ball_size: int,
        stride: int,
        depth: int,
        coord_dim: int,
        use_rotation: bool,
    ):
        super().__init__()

        self.unpool = BallUnpooling(dim, stride, coord_dim)
        self.blocks = nn.ModuleList(
            [
                BallTransformerBlock(dim, n_heads, ball_size, coord_dim)
                for _ in range(depth)
            ]
        )

        self.use_rotation = use_rotation
        self.rotate_pattern = [i % 2 == 1 for i in range(depth)]

    def forward(self, node: TreeNode) -> TreeNode:
        node = self.unpool(node)

        if self.use_rotation and node.tree_idx_rot is not None:
            inv_perm = torch.argsort(node.tree_idx_rot)

        for rotate, block in zip(self.rotate_pattern, self.blocks):
            if rotate and self.use_rotation and node.tree_idx_rot is not None:
                x_rot = node.x[node.tree_idx_rot]
                pos_rot = node.pos[node.tree_idx_rot]
                node.x = block(x_rot, pos_rot)[inv_perm]
            else:
                node.x = block(node.x, node.pos)

        return node


class CrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, out_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.q_proj = nn.Linear(q_dim, out_dim)
        self.k_proj = nn.Linear(kv_dim, out_dim)
        self.v_proj = nn.Linear(kv_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        Q = rearrange(self.q_proj(q), "n (h d) -> h n d", h=self.n_heads)
        K = rearrange(self.k_proj(kv), "m (h d) -> h m d", h=self.n_heads)
        V = rearrange(self.v_proj(kv), "m (h d) -> h m d", h=self.n_heads)

        out = F.scaled_dot_product_attention(Q, K, V)
        out = rearrange(out, "h n d -> n (h d)")

        return self.out_proj(out)


class GeometricEmbedding(nn.Module):
    def __init__(self, coord_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * coord_dim + 3, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, pos: torch.Tensor, ball_size: int) -> torch.Tensor:
        n_balls = pos.shape[0] // ball_size
        pos_balls = pos.view(n_balls, ball_size, -1)

        centers = pos_balls.mean(dim=1)
        delta = pos_balls - centers.unsqueeze(1)

        dists = delta.norm(dim=-1)
        mean_dist = dists.mean(dim=1, keepdim=True)
        dist_var = dists.var(dim=1, keepdim=True)
        count = torch.full((n_balls, 1), ball_size, device=pos.device, dtype=pos.dtype)

        cov = torch.einsum("bni,bnj->bij", delta, delta) / ball_size
        eigenvalues = torch.linalg.eigvalsh(cov)

        geo = torch.cat([centers, mean_dist, dist_var, count, eigenvalues], dim=-1)
        geo = self.mlp(geo)

        return geo.repeat_interleave(ball_size, dim=0)


class TreeCoderEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        coord_dim: int,
        ball_sizes: List[int],
        strides: List[int],
        enc_depths: List[int],
        n_heads: int,
        rotate_angle: float,
    ):
        super().__init__()

        self.coord_dim = coord_dim
        self.ball_sizes = ball_sizes
        self.strides = strides
        self.rotate_angle = rotate_angle
        self.use_rotation = rotate_angle > 0

        n_levels = len(strides)
        dims = [out_channels * (2**i) for i in range(n_levels + 1)]

        self.input_proj = nn.Linear(in_channels, out_channels)
        self.geo_embed = GeometricEmbedding(coord_dim, out_channels)

        self.layers = nn.ModuleList()
        for i in range(n_levels):
            self.layers.append(
                EncoderLayer(
                    dim=dims[i],
                    n_heads=n_heads,
                    ball_size=ball_sizes[i],
                    stride=strides[i],
                    depth=enc_depths[i],
                    coord_dim=coord_dim,
                    use_rotation=self.use_rotation,
                )
            )

        self.bottleneck = nn.ModuleList(
            [
                BallTransformerBlock(dims[-1], n_heads, ball_sizes[-1], coord_dim)
                for _ in range(enc_depths[-1])
            ]
        )
        self.bottleneck_rotate = [i % 2 == 1 for i in range(enc_depths[-1])]

        self.to_latent = CrossAttention(
            q_dim=coord_dim, kv_dim=dims[-1], out_dim=out_channels, n_heads=n_heads
        )
        self.latent_pos_proj = nn.Linear(coord_dim, out_channels)

    def forward(
        self,
        x_coord: torch.Tensor,
        pndata: torch.Tensor,
        latent_tokens_coord: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = pndata.shape
        device = pndata.device

        if x_coord.dim() == 2:
            x_coord = x_coord.unsqueeze(0).expand(B, -1, -1)

        outputs = []
        for b in range(B):
            pos = x_coord[b]
            feat = pndata[b]
            batch_idx = torch.zeros(N, dtype=torch.long, device=device)

            tree_idx, tree_mask, rot_indices = build_balltree_with_rotations(
                pos, batch_idx, self.strides, self.ball_sizes, self.rotate_angle
            )

            x = self.input_proj(feat)
            x = x[tree_idx]
            pos_tree = pos[tree_idx]
            batch_idx_tree = batch_idx[tree_idx]

            x = x + self.geo_embed(pos_tree, self.ball_sizes[0])

            node = TreeNode(
                x=x,
                pos=pos_tree,
                batch_idx=batch_idx_tree,
                tree_idx=tree_idx,
                tree_mask=tree_mask,
                tree_idx_rot=rot_indices[0] if rot_indices else None,
            )

            for i, layer in enumerate(self.layers):
                node.tree_idx_rot = (
                    rot_indices[i] if rot_indices and i < len(rot_indices) else None
                )
                node = layer(node)

            bn_rot_idx = (
                rot_indices[len(self.layers)]
                if rot_indices and len(rot_indices) > len(self.layers)
                else None
            )
            if bn_rot_idx is not None:
                inv_perm = torch.argsort(bn_rot_idx)

            for rotate, block in zip(self.bottleneck_rotate, self.bottleneck):
                if rotate and bn_rot_idx is not None:
                    node.x = block(node.x[bn_rot_idx], node.pos[bn_rot_idx])[inv_perm]
                else:
                    node.x = block(node.x, node.pos)

            latent_pos = (
                latent_tokens_coord
                if latent_tokens_coord.dim() == 2
                else latent_tokens_coord[b]
            )
            q = self.latent_pos_proj(latent_pos)
            z = self.to_latent(q, node.x)

            outputs.append(z)

        return torch.stack(outputs, dim=0)


class TreeCoderDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        coord_dim: int,
        ball_sizes: List[int],
        strides: List[int],
        dec_depths: List[int],
        n_heads: int,
        rotate_angle: float,
    ):
        super().__init__()

        self.coord_dim = coord_dim
        self.ball_sizes = ball_sizes
        self.strides = strides
        self.rotate_angle = rotate_angle
        self.use_rotation = rotate_angle > 0

        n_levels = len(strides)
        dims = [in_channels * (2**i) for i in range(n_levels + 1)]
        dims_reversed = dims[::-1]

        self.from_latent = CrossAttention(
            q_dim=coord_dim,
            kv_dim=in_channels,
            out_dim=dims_reversed[0],
            n_heads=n_heads,
        )

        self.layers = nn.ModuleList()
        for i in range(n_levels):
            self.layers.append(
                DecoderLayer(
                    dim=dims_reversed[i],
                    n_heads=n_heads,
                    ball_size=ball_sizes[-(i + 1)],
                    stride=strides[-(i + 1)],
                    depth=dec_depths[-(i + 1)],
                    coord_dim=coord_dim,
                    use_rotation=self.use_rotation,
                )
            )

        self.geo_embed = GeometricEmbedding(coord_dim, dims_reversed[-1])

        self.output_proj = nn.Sequential(
            nn.Linear(dims_reversed[-1] + coord_dim, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels),
        )

    def forward(
        self,
        latent_tokens_coord: torch.Tensor,
        rndata: torch.Tensor,
        query_coord: torch.Tensor,
    ) -> torch.Tensor:
        B, N_q, _ = (
            query_coord.shape
            if query_coord.dim() == 3
            else (rndata.shape[0], query_coord.shape[0], query_coord.shape[1])
        )
        device = rndata.device

        if query_coord.dim() == 2:
            query_coord = query_coord.unsqueeze(0).expand(B, -1, -1)
        if latent_tokens_coord.dim() == 2:
            latent_tokens_coord = latent_tokens_coord.unsqueeze(0).expand(B, -1, -1)

        N_q = query_coord.shape[1]

        outputs = []
        for b in range(B):
            query_pos = query_coord[b]
            latent_pos = latent_tokens_coord[b]
            latent_feat = rndata[b]
            batch_idx = torch.zeros(N_q, dtype=torch.long, device=device)

            tree_idx, tree_mask, rot_indices = build_balltree_with_rotations(
                query_pos, batch_idx, self.strides, self.ball_sizes, self.rotate_angle
            )

            query_pos_tree = query_pos[tree_idx]

            hierarchy = self._build_coarse_hierarchy(query_pos_tree)

            coarse_pos = hierarchy[-1]
            coarse_x = self.from_latent(coarse_pos, latent_feat)

            node = self._init_decoder_nodes(coarse_x, hierarchy, tree_idx, tree_mask)

            for i, layer in enumerate(self.layers):
                rot_idx = (
                    rot_indices[-(i + 1)]
                    if rot_indices and len(rot_indices) > i
                    else None
                )
                node.tree_idx_rot = rot_idx
                if node.children is not None:
                    node.children.tree_idx_rot = rot_idx
                node = layer(node)

            node.x = node.x + self.geo_embed(node.pos, self.ball_sizes[0])

            centers = reduce(node.pos, "(n s) d -> n d", "mean", s=self.ball_sizes[0])
            rel_pos = node.pos - centers.repeat_interleave(self.ball_sizes[0], dim=0)

            out = self.output_proj(torch.cat([node.x, rel_pos], dim=-1))

            out_valid = out[tree_mask]
            out_final = out_valid[torch.argsort(tree_idx[tree_mask])]

            outputs.append(out_final)

        return torch.stack(outputs, dim=0)

    def _build_coarse_hierarchy(self, pos: torch.Tensor) -> List[torch.Tensor]:
        hierarchy = [pos]
        current = pos

        for stride in self.strides:
            n = current.shape[0] // stride
            if n == 0:
                n = 1
            current = reduce(current[: n * stride], "(n s) d -> n d", "mean", s=stride)
            hierarchy.append(current)

        return hierarchy

    def _init_decoder_nodes(
        self,
        coarse_x: torch.Tensor,
        hierarchy: List[torch.Tensor],
        tree_idx: torch.Tensor,
        tree_mask: torch.Tensor,
    ) -> TreeNode:
        n_levels = len(self.strides)
        nodes = []

        for level in range(n_levels + 1):
            pos = hierarchy[n_levels - level]
            n = pos.shape[0]

            if level == 0:
                x = coarse_x
            else:
                x = torch.zeros(
                    n, coarse_x.shape[-1] // (2**level), device=coarse_x.device
                )

            batch_idx = torch.zeros(n, dtype=torch.long, device=pos.device)

            node = TreeNode(
                x=x,
                pos=pos,
                batch_idx=batch_idx,
                tree_idx=tree_idx[:n] if n <= tree_idx.shape[0] else tree_idx,
                tree_mask=tree_mask[:n] if n <= tree_mask.shape[0] else tree_mask,
                children=None,
            )
            nodes.append(node)

        for i in range(len(nodes) - 1):
            nodes[i].children = nodes[i + 1]

        return nodes[0]


class BOAT(nn.Module):
    def __init__(self, input_size: int, output_size: int, config):
        super().__init__()

        tc = config.args.treecoder
        coord_dim = tc.coord_dim

        self.input_size = input_size
        self.output_size = output_size
        self.coord_dim = coord_dim
        self.node_latent_size = tc.lifting_channels
        self.patch_size = config.args.transformer.patch_size

        latent_tokens_size = config.latent_tokens_size
        if coord_dim == 2:
            self.H, self.W = latent_tokens_size[0], latent_tokens_size[1]
            self.D = None
        else:
            self.H, self.W, self.D = latent_tokens_size

        self.encoder = TreeCoderEncoder(
            in_channels=input_size,
            out_channels=self.node_latent_size,
            coord_dim=coord_dim,
            ball_sizes=tc.ball_sizes,
            strides=tc.strides,
            enc_depths=tc.enc_depths,
            n_heads=tc.n_heads,
            rotate_angle=tc.rotate_angle,
        )

        self.processor = self._init_processor(config.args.transformer)

        self.decoder = TreeCoderDecoder(
            in_channels=self.node_latent_size,
            out_channels=output_size,
            coord_dim=coord_dim,
            ball_sizes=tc.ball_sizes,
            strides=tc.strides,
            dec_depths=tc.dec_depths,
            n_heads=tc.n_heads,
            rotate_angle=tc.rotate_angle,
        )

    def _init_processor(self, config) -> nn.Module:
        P = self.patch_size
        C = self.node_latent_size

        if self.coord_dim == 2:
            patch_vol = P * P
            n_patches_h, n_patches_w = self.H // P, self.W // P
            positions = torch.stack(
                torch.meshgrid(
                    torch.arange(n_patches_h, dtype=torch.float32),
                    torch.arange(n_patches_w, dtype=torch.float32),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
        else:
            patch_vol = P * P * P
            n_patches_h, n_patches_w, n_patches_d = (
                self.H // P,
                self.W // P,
                self.D // P,
            )
            positions = torch.stack(
                torch.meshgrid(
                    torch.arange(n_patches_h, dtype=torch.float32),
                    torch.arange(n_patches_w, dtype=torch.float32),
                    torch.arange(n_patches_d, dtype=torch.float32),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 3)

        self.register_buffer("patch_positions", positions)
        self.patch_linear = nn.Linear(patch_vol * C, patch_vol * C)
        self.positional_embedding_name = config.positional_embedding

        return Transformer(
            input_size=C * patch_vol, output_size=C * patch_vol, config=config
        )

    def _compute_absolute_embeddings(
        self, positions: torch.Tensor, embed_dim: int
    ) -> torch.Tensor:
        n_dims = positions.size(1)
        dim_per_axis = embed_dim // (2 * n_dims)
        freq_seq = torch.arange(
            dim_per_axis, dtype=torch.float32, device=positions.device
        )
        inv_freq = 1.0 / (10000 ** (freq_seq / dim_per_axis))
        sinusoid = positions[:, :, None] * inv_freq[None, None, :]
        return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1).view(
            positions.size(0), -1
        )

    def encode(
        self,
        x_coord: torch.Tensor,
        pndata: torch.Tensor,
        latent_tokens_coord: torch.Tensor,
        encoder_nbrs: Optional[list] = None,
    ) -> torch.Tensor:
        return self.encoder(x_coord, pndata, latent_tokens_coord)

    def process(
        self, rndata: torch.Tensor, condition: Optional[float] = None
    ) -> torch.Tensor:
        B, N, C = rndata.shape
        P = self.patch_size

        if self.coord_dim == 2:
            H, W = self.H, self.W
            ph, pw = H // P, W // P

            x = rndata.view(B, H, W, C)
            x = x.view(B, ph, P, pw, P, C)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.view(B, ph * pw, P * P * C)
        else:
            H, W, D = self.H, self.W, self.D
            ph, pw, pd = H // P, W // P, D // P

            x = rndata.view(B, H, W, D, C)
            x = x.view(B, ph, P, pw, P, pd, P, C)
            x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
            x = x.view(B, ph * pw * pd, P * P * P * C)

        x = self.patch_linear(x)

        if self.positional_embedding_name == "absolute":
            pos_emb = self._compute_absolute_embeddings(
                self.patch_positions, x.shape[-1]
            )
            x = x + pos_emb
            rel_pos = None
        else:
            rel_pos = self.patch_positions

        x = self.processor(x, condition=condition, relative_positions=rel_pos)

        if self.coord_dim == 2:
            x = x.view(B, ph, pw, P, P, C)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.view(B, H * W, C)
        else:
            x = x.view(B, ph, pw, pd, P, P, P, C)
            x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
            x = x.view(B, H * W * D, C)

        return x

    def decode(
        self,
        latent_tokens_coord: torch.Tensor,
        rndata: torch.Tensor,
        query_coord: torch.Tensor,
        decoder_nbrs: Optional[list] = None,
    ) -> torch.Tensor:
        return self.decoder(latent_tokens_coord, rndata, query_coord)

    def forward(
        self,
        latent_tokens_coord: torch.Tensor,
        xcoord: torch.Tensor,
        pndata: torch.Tensor,
        query_coord: Optional[torch.Tensor] = None,
        encoder_nbrs: Optional[list] = None,
        decoder_nbrs: Optional[list] = None,
        condition: Optional[float] = None,
    ) -> torch.Tensor:
        rndata = self.encode(xcoord, pndata, latent_tokens_coord, encoder_nbrs)
        rndata = self.process(rndata, condition)

        if query_coord is None:
            query_coord = xcoord

        return self.decode(latent_tokens_coord, rndata, query_coord, decoder_nbrs)

    def autoregressive_predict(
        self,
        x_batch: torch.Tensor,
        time_indices: np.ndarray,
        t_values: np.ndarray,
        stats: Dict,
        stepper_mode: str = "output",
        latent_tokens_coord: Optional[torch.Tensor] = None,
        fixed_coord: Optional[torch.Tensor] = None,
        encoder_nbrs: Optional[List] = None,
        decoder_nbrs: Optional[List] = None,
        use_conditional_norm: bool = False,
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = x_batch.shape
        num_timesteps = len(time_indices)
        predictions = []

        u_mean = stats["u"]["mean"].to(x_batch.device)
        u_std = stats["u"]["std"].to(x_batch.device)
        u_dim = u_mean.shape[0]

        c_dim = stats["c"]["mean"].shape[0] if "c" in stats else 0
        c_features = x_batch[..., u_dim : u_dim + c_dim] if c_dim > 0 else None

        current_u = x_batch[..., :u_dim]

        for idx in range(1, num_timesteps):
            t_in, t_out = time_indices[idx - 1], time_indices[idx]
            start_time = t_values[t_in]
            time_diff = t_values[t_out] - t_values[t_in]

            start_norm = (start_time - stats["start_time"]["mean"]) / stats[
                "start_time"
            ]["std"]
            diff_norm = (time_diff - stats["time_diffs"]["mean"]) / stats["time_diffs"][
                "std"
            ]

            t_start = torch.full(
                (batch_size, num_nodes, 1),
                start_norm,
                dtype=x_batch.dtype,
                device=x_batch.device,
            )
            t_diff = torch.full(
                (batch_size, num_nodes, 1),
                diff_norm,
                dtype=x_batch.dtype,
                device=x_batch.device,
            )

            inputs = [current_u]
            if c_features is not None:
                inputs.append(c_features)
            inputs.extend([t_start, t_diff])
            x_input = torch.cat(inputs, dim=-1)

            with torch.no_grad():
                pred = self.forward(
                    latent_tokens_coord=latent_tokens_coord,
                    xcoord=fixed_coord,
                    pndata=x_input,
                    encoder_nbrs=encoder_nbrs,
                    decoder_nbrs=decoder_nbrs,
                )

                if stepper_mode == "output":
                    pred_denorm = pred * u_std + u_mean
                elif stepper_mode == "residual":
                    res_mean, res_std = (
                        stats["res"]["mean"].to(pred.device),
                        stats["res"]["std"].to(pred.device),
                    )
                    pred_denorm = (current_u * u_std + u_mean) + (
                        pred * res_std + res_mean
                    )
                elif stepper_mode == "time_der":
                    der_mean, der_std = (
                        stats["der"]["mean"].to(pred.device),
                        stats["der"]["std"].to(pred.device),
                    )
                    pred_denorm = (current_u * u_std + u_mean) + time_diff * (
                        pred * der_std + der_mean
                    )

                predictions.append(pred_denorm)
                current_u = (pred_denorm - u_mean) / u_std

        return torch.stack(predictions, dim=1)
