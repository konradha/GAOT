import argparse
import logging
import random
from collections import OrderedDict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.model.gaot import GAOT
from src.model.boat import BOAT, BOATConfig
from src.model.layers.magno import MAGNOConfig
from src.model.layers.attn import TransformerConfig


VTKHDF_PATH = Path("/var/llm/sci-data/INS_2D_EAGLE-Drone-Wake.vtkhdf")
NEIGHBORS_PATH = Path("/var/llm/sci-data/eagle_neighbors.pt")
N_TIMESTEPS = 990
FIELD_NAMES = ["VX", "VY", "PS", "PG"]
MASK_KEY = "drone_mask"

SEED = 1
TRAIN_SIZE_DEFAULT = 150
VAL_SIZE_DEFAULT = 50
TEST_SIZE_DEFAULT = 50
MAX_TIME_DIFF = 14
TIME_STEP = 2
USE_TIME_NORM = True
SAMPLE_RATE = 0.1

LR = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 100
EVAL_EVERY = 2
MAX_LR = 1e-2
MIN_LR = 1e-5
FINAL_LR = 1e-5

SINGLE_PAIRS_PER_SAMPLE = 14
MAX_EVAL_ITERS = 2000

ALPHA = 0.1
NEIGHBOR_CACHE_SIZE = 64

MULTI_TAU_TRAIN_TAUS = [1, 3, 5, 15, 30]
MULTI_TAU_MAX_DIRECT = 30
MULTI_TAU_AR_STEP = 30
MULTI_TAU_EVAL_TAUS = [1, 5, 15, 30, 50, 250]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FIFOCache:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.cache = OrderedDict()

    def get(self, key):
        return self.cache.get(key)

    def put(self, key, value):
        if key in self.cache:
            return
        if len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self):
        self.cache.clear()


class MixScheduler:
    def __init__(self, optimizer, total_epochs, initial_lr, max_lr, min_lr, final_lr):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.final_lr = final_lr

        self.warmup_epochs = max(1, int(0.02 * total_epochs))
        self.cosine_epochs = int(0.90 * total_epochs)
        self.exp_epochs = total_epochs - self.warmup_epochs - self.cosine_epochs

        if self.exp_epochs <= 0:
            self.exp_epochs = 1
            self.cosine_epochs = max(1, self.cosine_epochs - 1)

        self.current_epoch = 0

    def get_lr(self):
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / max(1, self.warmup_epochs - 1)
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        elif self.current_epoch < self.warmup_epochs + self.cosine_epochs:
            epoch_in_cosine = self.current_epoch - self.warmup_epochs
            cosine_progress = epoch_in_cosine / self.cosine_epochs
            cosine_factor = (1 + np.cos(np.pi * cosine_progress)) / 2
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
        else:
            epoch_in_exp = self.current_epoch - self.warmup_epochs - self.cosine_epochs
            exp_progress = epoch_in_exp / max(1, self.exp_epochs - 1)
            lr = self.min_lr * ((self.final_lr / self.min_lr) ** exp_progress)
        return lr

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_epoch += 1
        return lr


def weighted_rms_loss(pred, target, alpha=ALPHA):
    vel_rms = torch.sqrt(torch.mean((pred[..., :2] - target[..., :2]) ** 2))
    pres_rms = torch.sqrt(torch.mean((pred[..., 2:] - target[..., 2:]) ** 2))
    return vel_rms + alpha * pres_rms


class EagleData:
    def __init__(self, vtkhdf_path: Path):
        with h5py.File(vtkhdf_path, "r") as f:
            self.node_counts = f["SampleNodeCount"][:].astype(np.int64)
            self.node_offsets = f["SampleNodeOffset"][:].astype(np.int64)
            self.points = f["VTKHDF/Points"][:, :2].astype(np.float32)

            self.vx = f["VTKHDF/PointData/VX"][:].astype(np.float32)
            self.vy = f["VTKHDF/PointData/VY"][:].astype(np.float32)
            self.ps = f["VTKHDF/PointData/PS"][:].astype(np.float32)
            self.pg = f["VTKHDF/PointData/PG"][:].astype(np.float32)
            self.mask = f[f"VTKHDF/PointData/{MASK_KEY}"][:].astype(np.float32)

        self.n_samples = len(self.node_counts)
        self.sample_base = self._compute_sample_base()
        self.coord_min = self.points.min(axis=0)
        self.coord_max = self.points.max(axis=0)
        self.coord_range = np.where(
            self.coord_max - self.coord_min == 0,
            np.ones_like(self.coord_max),
            self.coord_max - self.coord_min,
        )

        self.coords_cache = {}
        self.coords_norm_cache = {}
        self.mask_cache = {}

    def _compute_sample_base(self):
        cum_nodes = np.cumsum(self.node_counts)
        base = np.zeros_like(self.node_counts)
        base[1:] = cum_nodes[:-1] * N_TIMESTEPS
        return base

    def get_fields(self, sample_idx: int, t_idx: int):
        nc = int(self.node_counts[sample_idx])
        off = int(self.sample_base[sample_idx] + t_idx * nc)
        return np.stack(
            [
                self.vx[off : off + nc],
                self.vy[off : off + nc],
                self.ps[off : off + nc],
                self.pg[off : off + nc],
            ],
            axis=-1,
        ).astype(np.float32)

    def get_mask(self, sample_idx: int, t_idx: int):
        if (sample_idx, t_idx) in self.mask_cache:
            return self.mask_cache[(sample_idx, t_idx)]
        nc = int(self.node_counts[sample_idx])
        off = int(self.sample_base[sample_idx] + t_idx * nc)
        mask = self.mask[off : off + nc].astype(np.float32)
        self.mask_cache[(sample_idx, t_idx)] = mask
        return mask

    def get_coords(self, sample_idx: int):
        if sample_idx in self.coords_cache:
            return self.coords_cache[sample_idx]
        nc = int(self.node_counts[sample_idx])
        ns = int(self.node_offsets[sample_idx])
        coords = self.points[ns : ns + nc].copy()
        self.coords_cache[sample_idx] = coords
        return coords

    def get_coords_norm(self, sample_idx: int):
        if sample_idx in self.coords_norm_cache:
            return self.coords_norm_cache[sample_idx]
        coords = self.get_coords(sample_idx)
        coords_norm = 2.0 * (coords - self.coord_min) / self.coord_range - 1.0
        self.coords_norm_cache[sample_idx] = coords_norm.astype(np.float32)
        return self.coords_norm_cache[sample_idx]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_latent_grid(latent_tokens_size, device):
    h, w = latent_tokens_size
    x = torch.linspace(-1.0, 1.0, h, dtype=torch.float32, device=device)
    y = torch.linspace(-1.0, 1.0, w, dtype=torch.float32, device=device)
    grid = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1).reshape(-1, 2)
    return grid


def compute_stats(data, sample_indices, time_limit, sample_rate):
    n_samples = max(1, int(len(sample_indices) * sample_rate))
    samples = sample_indices[:n_samples]

    u_sums = np.zeros(4, dtype=np.float64)
    u_sq_sums = np.zeros(4, dtype=np.float64)
    u_count = 0

    res_sums = np.zeros(4, dtype=np.float64)
    res_sq_sums = np.zeros(4, dtype=np.float64)
    res_count = 0

    for sample_idx in samples:
        prev_vals = None
        for t_idx in range(time_limit):
            vals = data.get_fields(sample_idx, t_idx)
            u_sums += vals.sum(axis=0)
            u_sq_sums += (vals**2).sum(axis=0)
            u_count += vals.shape[0]

            if prev_vals is not None:
                residual = vals - prev_vals
                res_sums += residual.sum(axis=0)
                res_sq_sums += (residual**2).sum(axis=0)
                res_count += residual.shape[0]

            prev_vals = vals

    u_mean = u_sums / u_count
    u_std = np.sqrt(np.maximum(u_sq_sums / u_count - u_mean**2, 1e-12))
    u_std = np.maximum(u_std, 1e-6)

    res_mean = res_sums / res_count
    res_std = np.sqrt(np.maximum(res_sq_sums / res_count - res_mean**2, 1e-12))
    res_std = np.maximum(res_std, 1e-6)

    return (
        u_mean.astype(np.float32),
        u_std.astype(np.float32),
        res_mean.astype(np.float32),
        res_std.astype(np.float32),
    )


def compute_stats_multi_tau(data, sample_indices, train_taus, sample_rate):
    n_samples = max(1, int(len(sample_indices) * sample_rate))
    samples = sample_indices[:n_samples]
    max_tau = max(train_taus)

    u_sums = np.zeros(4, dtype=np.float64)
    u_sq_sums = np.zeros(4, dtype=np.float64)
    u_count = 0

    der_sums = {tau: np.zeros(4, dtype=np.float64) for tau in train_taus}
    der_sq_sums = {tau: np.zeros(4, dtype=np.float64) for tau in train_taus}
    der_count = {tau: 0 for tau in train_taus}

    for sample_idx in samples:
        cached = {}
        for t in range(N_TIMESTEPS):
            vals = data.get_fields(sample_idx, t)
            cached[t] = vals
            u_sums += vals.sum(axis=0)
            u_sq_sums += (vals**2).sum(axis=0)
            u_count += vals.shape[0]

        for t_in in range(N_TIMESTEPS - max_tau):
            for tau in train_taus:
                t_out = t_in + tau
                deriv = (cached[t_out] - cached[t_in]) / tau
                der_sums[tau] += deriv.sum(axis=0)
                der_sq_sums[tau] += (deriv**2).sum(axis=0)
                der_count[tau] += deriv.shape[0]

    u_mean = u_sums / u_count
    u_std = np.sqrt(np.maximum(u_sq_sums / u_count - u_mean**2, 1e-12))
    u_std = np.maximum(u_std, 1e-6)

    der_mean = {}
    der_std = {}
    for tau in train_taus:
        mean = der_sums[tau] / der_count[tau]
        std = np.sqrt(np.maximum(der_sq_sums[tau] / der_count[tau] - mean**2, 1e-12))
        std = np.maximum(std, 1e-6)
        der_mean[tau] = mean.astype(np.float32)
        der_std[tau] = std.astype(np.float32)

    tau_mean = float(np.mean(train_taus))
    tau_std = max(float(np.std(train_taus)), 1e-6)

    return {
        "u_mean": u_mean.astype(np.float32),
        "u_std": u_std.astype(np.float32),
        "der_mean": der_mean,
        "der_std": der_std,
        "tau_mean": tau_mean,
        "tau_std": tau_std,
    }


def build_time_pairs(max_time_diff, time_step):
    pairs = []
    for lag in range(time_step, max_time_diff + 1, time_step):
        for t_in in range(0, max_time_diff - lag + 1, time_step):
            pairs.append((t_in, t_in + lag))
    return pairs


def build_inputs_targets(data, stats, sample_idx, t_in, t_out, device):
    u_in_np = data.get_fields(sample_idx, t_in)
    u_out_np = data.get_fields(sample_idx, t_out)
    mask_np = data.get_mask(sample_idx, t_in)

    u_in = torch.from_numpy(u_in_np).to(device)
    u_out = torch.from_numpy(u_out_np).to(device)
    mask = torch.from_numpy(mask_np).to(device)

    u_mean = stats["u_mean"].to(device)
    u_std = stats["u_std"].to(device)
    res_mean = stats["res_mean"].to(device)
    res_std = stats["res_std"].to(device)

    u_in_norm = (u_in - u_mean) / u_std
    residual = u_out - u_in
    target = (residual - res_mean) / res_std

    inputs = torch.cat([u_in_norm, mask.unsqueeze(-1)], dim=-1)
    return inputs, target, u_in, res_mean, res_std


def build_inputs_targets_multi_tau(data, stats, sample_idx, t_in, tau, device):
    t_out = t_in + tau
    u_in_np = data.get_fields(sample_idx, t_in)
    u_out_np = data.get_fields(sample_idx, t_out)
    mask_np = data.get_mask(sample_idx, t_in)

    u_in = torch.from_numpy(u_in_np).to(device)
    u_out = torch.from_numpy(u_out_np).to(device)
    mask = torch.from_numpy(mask_np).to(device)

    u_mean = torch.tensor(stats["u_mean"], device=device)
    u_std = torch.tensor(stats["u_std"], device=device)

    der_mean = torch.tensor(stats["der_mean"][tau], device=device)
    der_std = torch.tensor(stats["der_std"][tau], device=device)

    u_in_norm = (u_in - u_mean) / u_std
    deriv = (u_out - u_in) / tau
    target = (deriv - der_mean) / der_std

    inputs = torch.cat([u_in_norm, mask.unsqueeze(-1)], dim=-1)
    return inputs, target, u_in, der_mean, der_std, tau


def load_gaot_neighbors(neighbors_path, all_indices, device):
    from tqdm import tqdm
    logger.info("Loading precomputed neighbors for GAOT...")
    nbrs = torch.load(neighbors_path, weights_only=False)
    latent_grid = torch.from_numpy(nbrs["latent_grid"].astype(np.float32)).to(device)
    coord_min, coord_max = nbrs["coord_min"], nbrs["coord_max"]

    encoder_nbrs, decoder_nbrs = {}, {}
    for sample_idx in tqdm(all_indices, desc="Caching neighbors on GPU", leave=False):
        encoder_nbrs[sample_idx] = [
            {k: torch.from_numpy(v).to(device) for k, v in scale.items()}
            for scale in nbrs["encoder_nbrs"][sample_idx]
        ]
        decoder_nbrs[sample_idx] = [
            {k: torch.from_numpy(v).to(device) for k, v in scale.items()}
            for scale in nbrs["decoder_nbrs"][sample_idx]
        ]

    logger.info("Neighbor loading complete.")
    return {
        "latent_grid": latent_grid,
        "coord_min": coord_min,
        "coord_max": coord_max,
        "encoder_nbrs": encoder_nbrs,
        "decoder_nbrs": decoder_nbrs,
    }


def forward_model_gaot(model, latent_grid, coords_norm, inputs, neighbor_cache, sample_idx):
    enc_nbrs, dec_nbrs = neighbor_cache.get(sample_idx)
    pred = model(
        latent_grid,
        coords_norm,
        inputs.unsqueeze(0),
        coords_norm,
        enc_nbrs,
        dec_nbrs,
    )
    return pred.squeeze(0)


def forward_model_boat(model, coords_raw, inputs, geometry_cache, sample_idx, tau_val=None):
    n = coords_raw.shape[0]
    if tau_val is None:
        tau_tensor = torch.ones(n, 1, device=coords_raw.device)
    else:
        tau_tensor = torch.full((n, 1), float(tau_val), device=coords_raw.device)

    batch_idx = torch.zeros(n, dtype=torch.long, device=coords_raw.device)
    pred = model.forward(coords_raw, inputs, batch_idx, tau=tau_tensor, sample_idx=sample_idx)
    return pred


def train_epoch(
    model,
    model_type,
    data,
    train_idx,
    time_pairs,
    time_limit,
    stats,
    optimizer,
    latent_grid,
    device,
    steps_per_epoch,
    rng,
    neighbor_cache,
):
    model.train()
    total_loss = 0.0

    for _ in range(steps_per_epoch):
        sample_idx = train_idx[rng.integers(len(train_idx))]

        if time_pairs is None:
            t_in = int(rng.integers(0, time_limit - 1))
            t_out = t_in + 1
        else:
            t_in, t_out = time_pairs[rng.integers(len(time_pairs))]

        inputs, target, _, _, _ = build_inputs_targets(
            data, stats, sample_idx, t_in, t_out, device
        )

        coords_raw = torch.from_numpy(data.get_coords(sample_idx)).to(device)
        coords_norm = torch.from_numpy(data.get_coords_norm(sample_idx)).to(device)

        optimizer.zero_grad()

        if model_type == "gaot":
            pred = forward_model_gaot(
                model, latent_grid, coords_norm, inputs, neighbor_cache, sample_idx
            )
        else:
            pred = forward_model_boat(
                model, coords_raw, inputs, neighbor_cache, sample_idx
            )

        loss = weighted_rms_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, steps_per_epoch)


@torch.no_grad()
def evaluate(
    model,
    model_type,
    data,
    indices,
    time_pairs,
    time_limit,
    stats,
    latent_grid,
    device,
    eval_iters,
    rng,
    neighbor_cache,
):
    model.eval()
    total_loss = 0.0
    total_rel_l2 = 0.0
    n = 0

    for _ in range(eval_iters):
        sample_idx = indices[rng.integers(len(indices))]
        if time_pairs is None:
            t_in = int(rng.integers(0, time_limit - 1))
            t_out = t_in + 1
        else:
            t_in, t_out = time_pairs[rng.integers(len(time_pairs))]

        inputs, target, u_in, res_mean, res_std = build_inputs_targets(
            data, stats, sample_idx, t_in, t_out, device
        )
        coords_raw = torch.from_numpy(data.get_coords(sample_idx)).to(device)
        coords_norm = torch.from_numpy(data.get_coords_norm(sample_idx)).to(device)

        if model_type == "gaot":
            pred = forward_model_gaot(
                model, latent_grid, coords_norm, inputs, neighbor_cache, sample_idx
            )
        else:
            pred = forward_model_boat(
                model, coords_raw, inputs, neighbor_cache, sample_idx
            )

        loss = weighted_rms_loss(pred, target)
        total_loss += loss.item()

        pred_res = pred * res_std + res_mean
        u_pred = u_in + pred_res
        u_out = u_in + target * res_std + res_mean

        rel_l2 = torch.sqrt(((u_pred - u_out) ** 2).sum()) / torch.sqrt((u_out**2).sum())
        total_rel_l2 += rel_l2.item()
        n += 1

    return total_loss / max(1, n), total_rel_l2 / max(1, n)


def train_epoch_multi_tau(
    model,
    model_type,
    data,
    train_idx,
    train_taus,
    stats,
    optimizer,
    latent_grid,
    device,
    steps_per_epoch,
    rng,
    neighbor_cache,
):
    model.train()
    total_loss = 0.0
    max_tau = max(train_taus)

    for _ in range(steps_per_epoch):
        sample_idx = train_idx[rng.integers(len(train_idx))]
        tau = int(rng.choice(train_taus))
        t_in = int(rng.integers(0, N_TIMESTEPS - max_tau))

        inputs, target, u_in, der_mean, der_std, tau_used = build_inputs_targets_multi_tau(
            data, stats, sample_idx, t_in, tau, device
        )

        coords_raw = torch.from_numpy(data.get_coords(sample_idx)).to(device)
        coords_norm = torch.from_numpy(data.get_coords_norm(sample_idx)).to(device)

        optimizer.zero_grad()

        if model_type == "gaot":
            pred = forward_model_gaot(
                model, latent_grid, coords_norm, inputs, neighbor_cache, sample_idx
            )
        else:
            pred = forward_model_boat(
                model, coords_raw, inputs, neighbor_cache, sample_idx, tau_val=tau
            )

        loss = weighted_rms_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, steps_per_epoch)


@torch.no_grad()
def evaluate_multi_tau(
    model,
    model_type,
    data,
    indices,
    train_taus,
    stats,
    latent_grid,
    device,
    eval_iters,
    rng,
    neighbor_cache,
):
    model.eval()
    total_loss = 0.0
    total_rel_l2 = 0.0
    n = 0
    max_tau = max(train_taus)

    for _ in range(eval_iters):
        sample_idx = indices[rng.integers(len(indices))]
        tau = int(rng.choice(train_taus))
        t_in = int(rng.integers(0, N_TIMESTEPS - max_tau))

        inputs, target, u_in, der_mean, der_std, tau_used = build_inputs_targets_multi_tau(
            data, stats, sample_idx, t_in, tau, device
        )

        coords_raw = torch.from_numpy(data.get_coords(sample_idx)).to(device)
        coords_norm = torch.from_numpy(data.get_coords_norm(sample_idx)).to(device)

        if model_type == "gaot":
            pred = forward_model_gaot(
                model, latent_grid, coords_norm, inputs, neighbor_cache, sample_idx
            )
        else:
            pred = forward_model_boat(
                model, coords_raw, inputs, neighbor_cache, sample_idx, tau_val=tau
            )

        loss = weighted_rms_loss(pred, target)
        total_loss += loss.item()

        pred_deriv = pred * der_std + der_mean
        u_pred = u_in + pred_deriv * tau
        u_out = u_in + target * der_std * tau + der_mean * tau

        rel_l2 = torch.sqrt(((u_pred - u_out) ** 2).sum()) / torch.sqrt((u_out**2).sum())
        total_rel_l2 += rel_l2.item()
        n += 1

    return total_loss / max(1, n), total_rel_l2 / max(1, n)


@torch.no_grad()
def ar_rollout_multi_tau(
    model,
    model_type,
    data,
    test_idx,
    stats,
    latent_grid,
    device,
    neighbor_cache,
    eval_taus,
    max_direct_tau,
    ar_step,
):
    model.eval()
    results = {t: [] for t in eval_taus}
    train_taus = list(stats["der_mean"].keys())

    def get_der_stats(tau):
        if tau in stats["der_mean"]:
            return (
                torch.tensor(stats["der_mean"][tau], device=device),
                torch.tensor(stats["der_std"][tau], device=device),
            )
        closest = min(train_taus, key=lambda t: abs(t - tau))
        return (
            torch.tensor(stats["der_mean"][closest], device=device),
            torch.tensor(stats["der_std"][closest], device=device),
        )

    u_mean = torch.tensor(stats["u_mean"], device=device)
    u_std = torch.tensor(stats["u_std"], device=device)

    for sample_idx in test_idx:
        coords_raw = torch.from_numpy(data.get_coords(sample_idx)).to(device)
        coords_norm = torch.from_numpy(data.get_coords_norm(sample_idx)).to(device)
        u_t0 = torch.from_numpy(data.get_fields(sample_idx, 0)).to(device)
        mask = torch.from_numpy(data.get_mask(sample_idx, 0)).to(device)

        for eval_tau in eval_taus:
            if eval_tau > N_TIMESTEPS - 1:
                continue

            u_gt = torch.from_numpy(data.get_fields(sample_idx, eval_tau)).to(device)

            if eval_tau <= max_direct_tau:
                u_in_norm = (u_t0 - u_mean) / u_std
                inputs = torch.cat([u_in_norm, mask.unsqueeze(-1)], dim=-1)

                if model_type == "gaot":
                    pred = forward_model_gaot(
                        model, latent_grid, coords_norm, inputs, neighbor_cache, sample_idx
                    )
                else:
                    pred = forward_model_boat(
                        model, coords_raw, inputs, neighbor_cache, sample_idx, tau_val=eval_tau
                    )

                der_mean, der_std = get_der_stats(eval_tau)
                pred_deriv = pred * der_std + der_mean
                u_pred = u_t0 + pred_deriv * eval_tau
            else:
                u_curr = u_t0.clone()
                remaining = eval_tau

                while remaining > 0:
                    step_tau = min(ar_step, remaining)
                    u_in_norm = (u_curr - u_mean) / u_std
                    inputs = torch.cat([u_in_norm, mask.unsqueeze(-1)], dim=-1)

                    if model_type == "gaot":
                        pred = forward_model_gaot(
                            model, latent_grid, coords_norm, inputs, neighbor_cache, sample_idx
                        )
                    else:
                        pred = forward_model_boat(
                            model, coords_raw, inputs, neighbor_cache, sample_idx, tau_val=step_tau
                        )

                    der_mean, der_std = get_der_stats(step_tau)
                    pred_deriv = pred * der_std + der_mean
                    u_curr = u_curr + pred_deriv * step_tau
                    remaining -= step_tau

                u_pred = u_curr

            rel_l2 = (
                torch.sqrt(((u_pred - u_gt) ** 2).sum())
                / torch.sqrt((u_gt**2).sum())
            ).item()
            results[eval_tau].append(rel_l2)

    return {
        t: {"mean": np.mean(vals), "std": np.std(vals)}
        for t, vals in results.items()
        if vals
    }


@torch.no_grad()
def predict_pair(
    model,
    model_type,
    data,
    stats,
    latent_grid,
    sample_idx,
    t_in,
    t_out,
    device,
    neighbor_cache,
):
    inputs, target, u_in, res_mean, res_std = build_inputs_targets(
        data, stats, sample_idx, t_in, t_out, device
    )
    coords_raw = torch.from_numpy(data.get_coords(sample_idx)).to(device)
    coords_norm = torch.from_numpy(data.get_coords_norm(sample_idx)).to(device)

    if model_type == "gaot":
        pred_norm = forward_model_gaot(
            model, latent_grid, coords_norm, inputs, neighbor_cache, sample_idx
        )
    else:
        pred_norm = forward_model_boat(
            model, coords_raw, inputs, neighbor_cache, sample_idx
        )

    pred_res = pred_norm * res_std + res_mean
    u_pred = u_in + pred_res
    u_out = u_in + target * res_std + res_mean

    return (
        u_pred.detach().cpu().numpy(),
        u_out.detach().cpu().numpy(),
        coords_raw.detach().cpu().numpy(),
    )


def plot_losses(train_epochs, train_losses, val_epochs, val_losses, out_path):
    plt.figure(figsize=(8, 6))
    plt.plot(train_epochs, train_losses, label="train")
    if val_epochs:
        plt.plot(val_epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error_maps(coords, abs_err, field_names, out_path):
    n_fields = abs_err.shape[1]
    fig, axes = plt.subplots(1, n_fields, figsize=(4 * n_fields, 4))
    if n_fields == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=abs_err[:, idx],
            s=2,
            cmap="magma",
        )
        ax.set_title(field_names[idx])
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_gaot_config():
    class Config:
        pass

    config = Config()
    config.latent_tokens_size = (64, 64)
    config.args = Config()
    config.args.magno = MAGNOConfig()
    config.args.transformer = TransformerConfig()
    return config


def build_boat_config(ball_sizes=None, strides=None):
    cfg = BOATConfig()
    if ball_sizes is not None:
        cfg.ball_sizes = ball_sizes
    if strides is not None:
        cfg.strides = strides
    return cfg


def parse_int_list(value):
    if value is None:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def split_indices(n_samples):
    total_default = TRAIN_SIZE_DEFAULT + VAL_SIZE_DEFAULT + TEST_SIZE_DEFAULT
    if n_samples >= total_default:
        train_size = TRAIN_SIZE_DEFAULT
        val_size = VAL_SIZE_DEFAULT
        test_size = TEST_SIZE_DEFAULT
    else:
        train_ratio = TRAIN_SIZE_DEFAULT / total_default
        val_ratio = VAL_SIZE_DEFAULT / total_default
        train_size = max(1, int(n_samples * train_ratio))
        val_size = max(1, int(n_samples * val_ratio))
        test_size = n_samples - train_size - val_size

        if test_size < 1:
            test_size = 1
            if val_size > 1:
                val_size -= 1
            elif train_size > 1:
                train_size -= 1

        while train_size + val_size + test_size > n_samples:
            if train_size >= val_size and train_size > 1:
                train_size -= 1
            elif val_size > 1:
                val_size -= 1
            elif test_size > 1:
                test_size -= 1
            else:
                break

    train_idx = list(range(0, train_size))
    val_idx = list(range(train_size, train_size + val_size))
    test_idx = list(range(n_samples - test_size, n_samples))
    return train_idx, val_idx, test_idx


def init_wandb(args, config):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping wandb logging.")
        return None

    mode = "offline" if args.wandb_offline else "online"
    try:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config,
            mode=mode,
        )
        return run
    except Exception as exc:
        logger.warning(f"wandb init failed: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gaot", "boat"], default="gaot")
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="single")
    parser.add_argument("--boat-ball-sizes", type=str, default=None)
    parser.add_argument("--boat-strides", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gaot-eagle")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Model: {args.model.upper()}, Mode: {args.mode}")
    logger.info(f"Device: {device}")

    data = EagleData(VTKHDF_PATH)
    train_idx, val_idx, test_idx = split_indices(data.n_samples)
    logger.info(
        f"Samples: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    is_multi = args.mode == "multi"

    if is_multi:
        steps_per_epoch = len(train_idx) * len(MULTI_TAU_TRAIN_TAUS) * 10
        stats = compute_stats_multi_tau(data, train_idx, MULTI_TAU_TRAIN_TAUS, SAMPLE_RATE)
        logger.info(f"u_mean: {stats['u_mean']}")
        logger.info(f"u_std: {stats['u_std']}")
        logger.info(f"tau_mean: {stats['tau_mean']:.2f}, tau_std: {stats['tau_std']:.2f}")
        for tau in MULTI_TAU_TRAIN_TAUS:
            logger.info(f"  tau={tau}: der_mean={stats['der_mean'][tau]}, der_std={stats['der_std'][tau]}")
    else:
        time_limit = N_TIMESTEPS
        time_pairs = None
        steps_per_epoch = len(train_idx) * min(SINGLE_PAIRS_PER_SAMPLE, time_limit - 1)
        u_mean, u_std, res_mean, res_std = compute_stats(
            data, train_idx, time_limit, SAMPLE_RATE
        )
        stats = {
            "u_mean": torch.tensor(u_mean, dtype=torch.float32),
            "u_std": torch.tensor(u_std, dtype=torch.float32),
            "res_mean": torch.tensor(res_mean, dtype=torch.float32),
            "res_std": torch.tensor(res_std, dtype=torch.float32),
        }
        logger.info(f"u_mean: {stats['u_mean']}")
        logger.info(f"u_std: {stats['u_std']}")
        logger.info(f"res_mean: {stats['res_mean']}")
        logger.info(f"res_std: {stats['res_std']}")

    input_size = 5
    output_size = 4

    all_indices = train_idx + val_idx + test_idx

    if args.model == "gaot":
        gaot_nbrs = load_gaot_neighbors(NEIGHBORS_PATH, all_indices, device)
        data.coord_min = gaot_nbrs["coord_min"]
        data.coord_max = gaot_nbrs["coord_max"]
        data.coord_range = np.where(
            data.coord_max - data.coord_min == 0,
            np.ones_like(data.coord_max),
            data.coord_max - data.coord_min,
        )
        data.coords_norm_cache.clear()
        latent_grid = gaot_nbrs["latent_grid"]
        neighbor_cache = FIFOCache(len(all_indices) + 16)
        for sample_idx in all_indices:
            neighbor_cache.put(
                sample_idx,
                (gaot_nbrs["encoder_nbrs"][sample_idx], gaot_nbrs["decoder_nbrs"][sample_idx])
            )
        config = build_gaot_config()
        model = GAOT(input_size=input_size, output_size=output_size, config=config)
    else:
        ball_sizes = parse_int_list(args.boat_ball_sizes)
        strides = parse_int_list(args.boat_strides)
        config = build_boat_config(ball_sizes=ball_sizes, strides=strides)
        model = BOAT(input_size=input_size, output_size=output_size, config=config)
        latent_grid = None
        neighbor_cache = FIFOCache(NEIGHBOR_CACHE_SIZE)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = MixScheduler(optimizer, EPOCHS, LR, MAX_LR, MIN_LR, FINAL_LR)

    run = init_wandb(
        args,
        {
            "model": args.model,
            "mode": args.mode,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_lr": MAX_LR,
            "min_lr": MIN_LR,
            "final_lr": FINAL_LR,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "input_size": input_size,
            "output_size": output_size,
            "steps_per_epoch": steps_per_epoch,
        },
    )

    ckpt_dir = Path(".ckpt/baseline_unified") / f"{args.model}_{args.mode}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loss_dir = Path(".loss/baseline_unified")
    loss_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(".result/baseline_unified") / f"{args.model}_{args.mode}"
    result_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_state = None

    train_losses = []
    train_epochs_list = []
    val_losses = []
    val_epochs_list = []

    rng = np.random.default_rng(SEED)

    logger.info("=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)

    for epoch in range(EPOCHS):
        lr = scheduler.get_lr()

        if is_multi:
            train_loss = train_epoch_multi_tau(
                model,
                args.model,
                data,
                train_idx,
                MULTI_TAU_TRAIN_TAUS,
                stats,
                optimizer,
                latent_grid,
                device,
                steps_per_epoch,
                rng,
                neighbor_cache,
            )
        else:
            train_loss = train_epoch(
                model,
                args.model,
                data,
                train_idx,
                time_pairs,
                time_limit,
                stats,
                optimizer,
                latent_grid,
                device,
                steps_per_epoch,
                rng,
                neighbor_cache,
            )

        scheduler.step()
        train_losses.append(train_loss)
        train_epochs_list.append(epoch + 1)

        if (epoch + 1) % EVAL_EVERY == 0:
            if is_multi:
                eval_iters = min(MAX_EVAL_ITERS, len(val_idx) * len(MULTI_TAU_TRAIN_TAUS) * 5)
                val_loss, val_rel = evaluate_multi_tau(
                    model,
                    args.model,
                    data,
                    val_idx,
                    MULTI_TAU_TRAIN_TAUS,
                    stats,
                    latent_grid,
                    device,
                    eval_iters,
                    rng,
                    neighbor_cache,
                )
            else:
                eval_iters = min(
                    MAX_EVAL_ITERS,
                    len(val_idx) * min(SINGLE_PAIRS_PER_SAMPLE, time_limit - 1),
                )
                val_loss, val_rel = evaluate(
                    model,
                    args.model,
                    data,
                    val_idx,
                    time_pairs,
                    time_limit,
                    stats,
                    latent_grid,
                    device,
                    eval_iters,
                    rng,
                    neighbor_cache,
                )

            val_losses.append(val_loss)
            val_epochs_list.append(epoch + 1)

            logger.info(
                f"Epoch {epoch + 1:3d} | lr={lr:.2e} | train={train_loss:.4f} | val={val_loss:.4f} | rel_l2={val_rel * 100:.2f}%"
            )
            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "stats": stats,
                }
                torch.save(best_state, ckpt_dir / "best.pt")
                logger.info("  -> saved best")
        else:
            logger.info(f"Epoch {epoch + 1:3d} | lr={lr:.2e} | train={train_loss:.4f}")

        if run is not None:
            payload = {"train_loss": train_loss, "lr": lr, "epoch": epoch + 1}
            if val_epochs_list and val_epochs_list[-1] == epoch + 1:
                payload["val_loss"] = val_losses[-1]
            run.log(payload)

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    logger.info("=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    if is_multi:
        eval_iters = min(MAX_EVAL_ITERS, len(test_idx) * len(MULTI_TAU_TRAIN_TAUS) * 5)
        test_loss, test_rel = evaluate_multi_tau(
            model,
            args.model,
            data,
            test_idx,
            MULTI_TAU_TRAIN_TAUS,
            stats,
            latent_grid,
            device,
            eval_iters,
            rng,
            neighbor_cache,
        )
        logger.info(f"Test loss: {test_loss:.4f}, rel_l2: {test_rel * 100:.2f}%")

        logger.info("AR Rollout on test set:")
        rollout = ar_rollout_multi_tau(
            model,
            args.model,
            data,
            test_idx,
            stats,
            latent_grid,
            device,
            neighbor_cache,
            MULTI_TAU_EVAL_TAUS,
            MULTI_TAU_MAX_DIRECT,
            MULTI_TAU_AR_STEP,
        )
        logger.info(f"{'tau':<8} {'Method':<10} {'Rel L2':<12} {'Std':<12}")
        logger.info("-" * 44)
        for tau in MULTI_TAU_EVAL_TAUS:
            if tau in rollout:
                r = rollout[tau]
                method = "direct" if tau <= MULTI_TAU_MAX_DIRECT else f"AR({MULTI_TAU_AR_STEP})"
                logger.info(f"{tau:<8} {method:<10} {r['mean'] * 100:.2f}%       {r['std'] * 100:.2f}%")

        if run is not None:
            run.log({"test_loss": test_loss, "test_rel_l2": test_rel})
            for tau, r in rollout.items():
                run.log({f"rollout_tau{tau}_mean": r["mean"], f"rollout_tau{tau}_std": r["std"]})
    else:
        eval_iters = min(
            MAX_EVAL_ITERS,
            len(test_idx) * min(SINGLE_PAIRS_PER_SAMPLE, time_limit - 1),
        )
        test_loss, test_rel = evaluate(
            model,
            args.model,
            data,
            test_idx,
            time_pairs,
            time_limit,
            stats,
            latent_grid,
            device,
            eval_iters,
            rng,
            neighbor_cache,
        )
        logger.info(f"Test loss: {test_loss:.4f}, rel_l2: {test_rel * 100:.2f}%")
        if run is not None:
            run.log({"test_loss": test_loss, "test_rel_l2": test_rel})

    loss_path = loss_dir / f"{args.model}_{args.mode}.png"
    plot_losses(train_epochs_list, train_losses, val_epochs_list, val_losses, loss_path)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
