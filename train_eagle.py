import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy
import os
import json

from src.model.gaot import GAOT
from src.model.layers.magno import MAGNOConfig
from src.model.layers.attn import TransformerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def log_info(msg):
    if is_main_process():
        logger.info(msg)


class Config:
    pass


FIELD_NAMES = ["VX", "VY", "PS", "PG"]


def build_config():
    config = Config()
    config.latent_tokens_size = [64, 64]
    config.args = Config()
    config.args.magno = MAGNOConfig(
        coord_dim=2,
        radius=0.1,
        hidden_size=64,
        mlp_layers=3,
        lifting_channels=32,
        scales=[1.0, 2.0],
        use_attention=True,
        use_geoembed=True,
        embedding_method="statistical",
        transform_type="linear",
        precompute_edges=True,
    )
    config.args.transformer = TransformerConfig(
        patch_size=8,
        hidden_size=256,
        num_layers=3,
        positional_embedding="absolute",
    )
    return config


def build_training_config():
    config = Config()

    config.train_size = 200
    config.val_size = 50
    config.test_size = 50

    config.max_time_diff = 30
    config.time_step = 3
    config.pairs_per_sample = 30

    config.n_epochs = 500
    config.eval_every = 2

    config.lr = 1e-3
    config.max_lr = 1.25e-3  # 1e-2 <- eff batch is 8
    config.min_lr = 1e-5
    config.final_lr = 1e-5
    config.weight_decay = 1e-3

    config.warmup_fraction = 0.1  # 0.02 <- we use smaller epoch num
    config.cosine_fraction = 0.90

    config.accumulation_steps = 8

    config.eval_tau_values = [1, 3, 6, 9, 15, 30, 51, 90, 180, 249]
    config.direct_tau_max = 30

    config.seed = 1

    return config


class MixScheduler:
    # training strategy aligned to GAOT suggestions
    def __init__(
        self,
        optimizer,
        total_epochs,
        lr,
        max_lr,
        min_lr,
        final_lr,
        warmup_fraction=0.02,
        cosine_fraction=0.90,
    ):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.lr = lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.final_lr = final_lr

        self.warmup_epochs = max(1, int(warmup_fraction * total_epochs))
        self.cosine_epochs = int(cosine_fraction * total_epochs)
        self.exp_epochs = total_epochs - self.warmup_epochs - self.cosine_epochs

        if self.exp_epochs <= 0:
            self.exp_epochs = 1
            self.cosine_epochs -= 1

        self.current_epoch = 0

    def get_lr(self):
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / max(1, self.warmup_epochs - 1)
            return self.lr + (self.max_lr - self.lr) * progress
        elif self.current_epoch < self.warmup_epochs + self.cosine_epochs:
            epoch_in_cosine = self.current_epoch - self.warmup_epochs
            cosine_progress = epoch_in_cosine / self.cosine_epochs
            cosine_factor = (1 + np.cos(np.pi * cosine_progress)) / 2
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
        else:
            epoch_in_exp = self.current_epoch - self.warmup_epochs - self.cosine_epochs
            exp_progress = epoch_in_exp / max(1, self.exp_epochs - 1)
            return self.min_lr * ((self.final_lr / self.min_lr) ** exp_progress)

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_epoch += 1
        return lr


def compute_global_stats(
    vtkhdf_path,
    sample_indices,
    n_timesteps=990,
    max_time_diff=30,
    time_step=3,
    n_samples_for_stats=50,
):
    log_info(f"Computing global statistics over full trajectory...")

    samples_to_use = sample_indices[: min(n_samples_for_stats, len(sample_indices))]

    with h5py.File(vtkhdf_path, "r") as f:
        node_counts = f["SampleNodeCount"][:]
        vx, vy = f["VTKHDF/PointData/VX"], f["VTKHDF/PointData/VY"]
        ps, pg = f["VTKHDF/PointData/PS"], f["VTKHDF/PointData/PG"]

        u_sums = np.zeros(4, dtype=np.float64)
        u_sq_sums = np.zeros(4, dtype=np.float64)
        u_count = 0

        der_sums = np.zeros(4, dtype=np.float64)
        der_sq_sums = np.zeros(4, dtype=np.float64)
        der_count = 0

        iterator = (
            tqdm(samples_to_use, desc="Stats") if is_main_process() else samples_to_use
        )
        for sample_idx in iterator:
            nc = int(node_counts[sample_idx])
            base = sum(int(node_counts[s]) * n_timesteps for s in range(sample_idx))

            prev_vals = None
            for t in range(0, n_timesteps, time_step):
                off = base + t * nc
                vals = np.stack(
                    [
                        vx[off : off + nc],
                        vy[off : off + nc],
                        ps[off : off + nc],
                        pg[off : off + nc],
                    ],
                    axis=-1,
                )

                u_sums += vals.sum(axis=0)
                u_sq_sums += (vals**2).sum(axis=0)
                u_count += nc

                if prev_vals is not None:
                    deriv = (vals - prev_vals) / time_step
                    der_sums += deriv.sum(axis=0)
                    der_sq_sums += (deriv**2).sum(axis=0)
                    der_count += nc

                prev_vals = vals

    u_mean = u_sums / u_count
    u_std = np.maximum(np.sqrt(u_sq_sums / u_count - u_mean**2), 1e-6)

    der_mean = der_sums / der_count
    der_std = np.maximum(np.sqrt(der_sq_sums / der_count - der_mean**2), 1e-6)

    max_valid_t = n_timesteps - 1 - max_time_diff
    all_start_times = np.arange(0, max_valid_t + 1, time_step, dtype=np.float32)
    all_lags = np.arange(time_step, max_time_diff + 1, time_step, dtype=np.float32)

    start_time_mean = float(all_start_times.mean())
    start_time_std = float(max(all_start_times.std(), 1e-6))
    time_diff_mean = float(all_lags.mean())
    time_diff_std = float(max(all_lags.std(), 1e-6))

    stats = {
        "u_mean": torch.tensor(u_mean, dtype=torch.float32),
        "u_std": torch.tensor(u_std, dtype=torch.float32),
        "der_mean": torch.tensor(der_mean, dtype=torch.float32),
        "der_std": torch.tensor(der_std, dtype=torch.float32),
        "start_time_mean": start_time_mean,
        "start_time_std": start_time_std,
        "time_diff_mean": time_diff_mean,
        "time_diff_std": time_diff_std,
        "n_timesteps": n_timesteps,
    }

    log_info(f"Stats computed from {len(samples_to_use)} samples, {u_count} points")
    log_info(f"  u_mean={u_mean}, u_std={u_std}")
    log_info(f"  der_mean={der_mean}, der_std={der_std}")
    log_info(
        f"  start_time~N({start_time_mean:.1f},{start_time_std:.1f}), "
        f"lag~N({time_diff_mean:.1f},{time_diff_std:.1f})"
    )

    return stats


def load_shared_data(
    vtkhdf_path, neighbors_path, all_indices, max_time_diff, time_step, n_timesteps=990
):
    log_info(
        f"Loading shared VTKHDF data for {len(all_indices)} samples (full trajectories)..."
    )

    with h5py.File(vtkhdf_path, "r") as f:
        node_offsets = f["SampleNodeOffset"][:].astype(np.int64)
        node_counts = f["SampleNodeCount"][:].astype(np.int64)
        points = f["VTKHDF/Points"][:, :2].astype(np.float32)

        vx_full = f["VTKHDF/PointData/VX"][:]
        vy_full = f["VTKHDF/PointData/VY"][:]
        ps_full = f["VTKHDF/PointData/PS"][:]
        pg_full = f["VTKHDF/PointData/PG"][:]
        mask_full = f["VTKHDF/PointData/drone_mask"][:]

    fields, masks, coords = {}, {}, {}
    iterator = (
        tqdm(all_indices, desc="Preloading samples")
        if is_main_process()
        else all_indices
    )

    for sample_idx in iterator:
        nc = int(node_counts[sample_idx])
        ns = int(node_offsets[sample_idx])
        base = sum(int(node_counts[s]) * n_timesteps for s in range(sample_idx))
        coords[sample_idx] = points[ns : ns + nc].copy()

        for t in range(0, n_timesteps, time_step):
            off = base + t * nc
            fields[(sample_idx, t)] = np.stack(
                [
                    vx_full[off : off + nc],
                    vy_full[off : off + nc],
                    ps_full[off : off + nc],
                    pg_full[off : off + nc],
                ],
                axis=-1,
            ).astype(np.float32)
            masks[(sample_idx, t)] = mask_full[off : off + nc].astype(np.float32)

    del vx_full, vy_full, ps_full, pg_full, mask_full

    log_info(f"Loaded {len(fields)} field snapshots across {len(all_indices)} samples")

    log_info("Loading neighbor graphs...")
    nbrs = torch.load(neighbors_path, weights_only=False)
    latent_grid = nbrs["latent_grid"].astype(np.float32)
    coord_min, coord_max = nbrs["coord_min"], nbrs["coord_max"]

    encoder_nbrs_all, decoder_nbrs_all = {}, {}
    for sample_idx in all_indices:
        encoder_nbrs_all[sample_idx] = nbrs["encoder_nbrs"][sample_idx]
        decoder_nbrs_all[sample_idx] = nbrs["decoder_nbrs"][sample_idx]

    return {
        "fields": fields,
        "masks": masks,
        "coords": coords,
        "latent_grid": latent_grid,
        "coord_min": coord_min,
        "coord_max": coord_max,
        "encoder_nbrs_all": encoder_nbrs_all,
        "decoder_nbrs_all": decoder_nbrs_all,
        "n_timesteps": n_timesteps,
    }


class EAGLEDataset(Dataset):
    """
    this is a sliding-window strategy
    For each sample:
    1. Pick random t_start ∈ [0, n_timesteps - max_time_diff - 1]
    2. Pick random lag ∈ [time_step, max_time_diff]
    3. t_in = t_start, t_out = t_start + lag
    """

    def __init__(
        self,
        shared_data,
        sample_indices,
        stats,
        max_time_diff=30,
        time_step=3,
        pairs_per_sample=30,
        n_timesteps=990,
    ):
        self.sample_indices = sample_indices
        self.max_time_diff = max_time_diff
        self.time_step = time_step
        self.pairs_per_sample = pairs_per_sample
        self.n_timesteps = n_timesteps

        self.u_mean = stats["u_mean"].numpy()
        self.u_std = stats["u_std"].numpy()
        self.der_mean = stats["der_mean"].numpy()
        self.der_std = stats["der_std"].numpy()
        self.start_time_mean = stats["start_time_mean"]
        self.start_time_std = stats["start_time_std"]
        self.time_diff_mean = stats["time_diff_mean"]
        self.time_diff_std = stats["time_diff_std"]

        self.fields = shared_data["fields"]
        self.masks = shared_data["masks"]
        self.coords = shared_data["coords"]
        self.coord_min = shared_data["coord_min"]
        self.coord_max = shared_data["coord_max"]

        self.max_t_start = n_timesteps - max_time_diff - 1
        self.valid_t_starts = np.arange(0, self.max_t_start + 1, time_step)
        self.valid_lags = np.arange(time_step, max_time_diff + 1, time_step)

    def __len__(self):
        return len(self.sample_indices) * self.pairs_per_sample

    def __getitem__(self, idx):
        sample_local = idx // self.pairs_per_sample
        sample_idx = self.sample_indices[sample_local]

        t_in = int(np.random.choice(self.valid_t_starts))
        lag = int(np.random.choice(self.valid_lags))
        t_out = t_in + lag

        coords = self.coords[sample_idx]
        u_in = self.fields[(sample_idx, t_in)]
        u_out = self.fields[(sample_idx, t_out)]
        mask = self.masks[(sample_idx, t_in)]
        nc = coords.shape[0]

        u_in_norm = (u_in - self.u_mean) / self.u_std

        du_dt = (u_out - u_in) / lag
        target = (du_dt - self.der_mean) / self.der_std

        start_time_norm = (float(t_in) - self.start_time_mean) / self.start_time_std
        time_diff_norm = (float(lag) - self.time_diff_mean) / self.time_diff_std

        time_feats = np.full(
            (nc, 2), [start_time_norm, time_diff_norm], dtype=np.float32
        )

        inputs = np.concatenate([u_in_norm, time_feats, mask[:, None]], axis=-1)

        coords_norm = (
            2.0 * (coords - self.coord_min) / (self.coord_max - self.coord_min) - 1.0
        )

        return (
            inputs.astype(np.float32),
            target.astype(np.float32),
            u_in_norm.astype(np.float32),
            coords_norm.astype(np.float32),
            mask.astype(np.float32),
            np.float32(lag),
            sample_idx,
        )


def collate_fn(batch):
    inputs, target, u_in_norm, coords, mask, dt, sample_idx = batch[0]
    return (
        torch.from_numpy(inputs).unsqueeze(0),
        torch.from_numpy(target).unsqueeze(0),
        torch.from_numpy(u_in_norm).unsqueeze(0),
        torch.from_numpy(coords),
        torch.from_numpy(mask),
        torch.tensor(dt, dtype=torch.float32),
        sample_idx,
    )


def create_dataloaders(
    shared_data,
    stats,
    train_idx,
    val_idx,
    test_idx,
    max_time_diff=30,
    time_step=3,
    pairs_per_sample=30,
    n_timesteps=990,
    world_size=1,
    rank=0,
):
    train_ds = EAGLEDataset(
        shared_data,
        train_idx,
        stats,
        max_time_diff,
        time_step,
        pairs_per_sample,
        n_timesteps,
    )
    val_ds = EAGLEDataset(
        shared_data,
        val_idx,
        stats,
        max_time_diff,
        time_step,
        pairs_per_sample,
        n_timesteps,
    )
    test_ds = EAGLEDataset(
        shared_data,
        test_idx,
        stats,
        max_time_diff,
        time_step,
        pairs_per_sample,
        n_timesteps,
    )

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        sampler=test_sampler,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    latent_grid = torch.from_numpy(shared_data["latent_grid"]).float()

    encoder_nbrs, decoder_nbrs = {}, {}
    for idx in train_idx + val_idx + test_idx:
        encoder_nbrs[idx] = [
            {k: torch.from_numpy(v) for k, v in scale.items()}
            for scale in shared_data["encoder_nbrs_all"][idx]
        ]
        decoder_nbrs[idx] = [
            {k: torch.from_numpy(v) for k, v in scale.items()}
            for scale in shared_data["decoder_nbrs_all"][idx]
        ]

    return (
        train_loader,
        val_loader,
        test_loader,
        latent_grid,
        encoder_nbrs,
        decoder_nbrs,
        train_sampler,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    latent_grid,
    encoder_nbrs,
    decoder_nbrs,
    device,
    world_size=1,
    sampler=None,
    epoch=0,
    accumulation_steps=8,
):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    total_loss = torch.zeros(1, device=device)
    n_samples = 0
    accumulated_loss = torch.zeros(1, device=device)

    pbar = tqdm(loader, desc="Train", leave=False, disable=not is_main_process())

    for step, (inputs, target, u_in_norm, coords, mask, dt, sample_idx) in enumerate(
        pbar
    ):
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)

        enc = [
            {k: v.to(device, non_blocking=True) for k, v in s.items()}
            for s in encoder_nbrs[sample_idx]
        ]
        dec = [
            {k: v.to(device, non_blocking=True) for k, v in s.items()}
            for s in decoder_nbrs[sample_idx]
        ]

        pred = model(latent_grid, coords, inputs, coords, enc, dec)
        loss = criterion(pred.squeeze(0), target.squeeze(0))

        loss_scaled = loss / accumulation_steps
        loss_scaled.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        accumulated_loss += loss.detach()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += accumulated_loss
            n_samples += accumulation_steps

            if is_main_process():
                avg_loss = accumulated_loss.item() / accumulation_steps
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

            accumulated_loss.zero_()

    if (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        remaining = (step + 1) % accumulation_steps
        total_loss += accumulated_loss
        n_samples += remaining

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        n_total = torch.tensor(n_samples, device=device)
        dist.all_reduce(n_total, op=dist.ReduceOp.SUM)
        return (total_loss / n_total).item()
    else:
        return (total_loss / max(n_samples, 1)).item()


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    latent_grid,
    encoder_nbrs,
    decoder_nbrs,
    device,
    stats,
    world_size=1,
):
    model.eval()

    total_loss = torch.zeros(1, device=device)
    total_rel = torch.zeros(1, device=device)
    per_field_rel = torch.zeros(4, device=device)
    n = torch.zeros(1, device=device)

    u_mean = stats["u_mean"].to(device)
    u_std = stats["u_std"].to(device)
    der_mean = stats["der_mean"].to(device)
    der_std = stats["der_std"].to(device)

    pbar = tqdm(loader, desc="Eval", leave=False, disable=not is_main_process())

    for inputs, target, u_in_norm, coords, mask, dt, sample_idx in pbar:
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        u_in_norm = u_in_norm.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        dt = dt.to(device, non_blocking=True)

        enc = [
            {k: v.to(device, non_blocking=True) for k, v in s.items()}
            for s in encoder_nbrs[sample_idx]
        ]
        dec = [
            {k: v.to(device, non_blocking=True) for k, v in s.items()}
            for s in decoder_nbrs[sample_idx]
        ]

        pred = model(latent_grid, coords, inputs, coords, enc, dec)

        total_loss += criterion(pred.squeeze(0), target.squeeze(0))

        pred_deriv = pred * der_std + der_mean
        pred_u_out = u_in_norm * u_std + u_mean + pred_deriv * dt

        target_deriv = target * der_std + der_mean
        tgt_u_out = u_in_norm * u_std + u_mean + target_deriv * dt

        diff = (pred_u_out - tgt_u_out).abs()
        total_rel += diff.sum() / tgt_u_out.abs().sum()

        for f in range(4):
            per_field_rel[f] += diff[..., f].sum() / tgt_u_out[..., f].abs().sum()
        n += 1

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_rel, op=dist.ReduceOp.SUM)
        dist.all_reduce(per_field_rel, op=dist.ReduceOp.SUM)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)

    n_val = n.item()
    return (
        (total_loss / n_val).item(),
        (total_rel / n_val).item(),
        (per_field_rel / n_val).cpu().numpy(),
    )


@torch.no_grad()
def autoregressive_rollout(
    model,
    shared_data,
    sample_indices,
    stats,
    latent_grid,
    encoder_nbrs,
    decoder_nbrs,
    device,
    time_step=3,
    eval_tau_values=None,
    direct_tau_max=30,
):
    model.eval()

    if eval_tau_values is None:
        eval_tau_values = [1, 3, 6, 9, 15, 30, 51, 90, 180, 249]

    eval_tau_values = [
        tau for tau in eval_tau_values if tau % time_step == 0 or tau == 1
    ]
    if 1 in eval_tau_values and time_step != 1:
        eval_tau_values.remove(1)

    u_mean = stats["u_mean"].to(device)
    u_std = stats["u_std"].to(device)
    der_mean = stats["der_mean"].to(device)
    der_std = stats["der_std"].to(device)
    start_time_mean = stats["start_time_mean"]
    start_time_std = stats["start_time_std"]
    time_diff_mean = stats["time_diff_mean"]
    time_diff_std = stats["time_diff_std"]

    coord_min = torch.tensor(
        shared_data["coord_min"], device=device, dtype=torch.float32
    )
    coord_max = torch.tensor(
        shared_data["coord_max"], device=device, dtype=torch.float32
    )

    n_timesteps = shared_data.get("n_timesteps", 990)

    results = {
        tau: {
            "rel_l1": [],
            "rel_l2": [],
            "rmse": [],
            "per_field_rel_l1": [],
            "per_field_rmse": [],
        }
        for tau in eval_tau_values
    }

    for sample_idx in tqdm(
        sample_indices, desc="Rollout", disable=not is_main_process()
    ):
        coords_np = shared_data["coords"][sample_idx]
        coords = torch.from_numpy(coords_np).to(device)
        coords_norm = 2.0 * (coords - coord_min) / (coord_max - coord_min) - 1.0
        nc = coords.shape[0]

        enc = [
            {k: v.to(device) for k, v in s.items()} for s in encoder_nbrs[sample_idx]
        ]
        dec = [
            {k: v.to(device) for k, v in s.items()} for s in decoder_nbrs[sample_idx]
        ]

        u_t0 = torch.from_numpy(shared_data["fields"][(sample_idx, 0)]).to(device)

        for tau in eval_tau_values:
            if tau > n_timesteps - 1:
                continue

            if (sample_idx, tau) not in shared_data["fields"]:
                continue

            u_gt = torch.from_numpy(shared_data["fields"][(sample_idx, tau)]).to(device)

            if tau <= direct_tau_max:
                u_curr = u_t0.clone()
                u_curr_norm = (u_curr - u_mean) / u_std

                start_time_norm = (0.0 - start_time_mean) / start_time_std
                time_diff_norm = (float(tau) - time_diff_mean) / time_diff_std

                mask = torch.from_numpy(shared_data["masks"][(sample_idx, 0)]).to(
                    device
                )
                time_feats = torch.full(
                    (nc, 2), 0.0, device=device, dtype=torch.float32
                )
                time_feats[:, 0] = start_time_norm
                time_feats[:, 1] = time_diff_norm

                inputs = torch.cat(
                    [u_curr_norm, time_feats, mask.unsqueeze(-1)], dim=-1
                ).unsqueeze(0)

                pred = model(latent_grid, coords_norm, inputs, coords_norm, enc, dec)
                pred_deriv = pred.squeeze(0) * der_std + der_mean
                u_pred = u_curr + pred_deriv * tau
            else:
                u_curr = u_t0.clone()
                n_steps = tau // time_step

                for step in range(n_steps):
                    t_curr = step * time_step

                    if (sample_idx, t_curr) in shared_data["masks"]:
                        mask = torch.from_numpy(
                            shared_data["masks"][(sample_idx, t_curr)]
                        ).to(device)
                    else:
                        mask = torch.from_numpy(
                            shared_data["masks"][(sample_idx, 0)]
                        ).to(device)

                    u_curr_norm = (u_curr - u_mean) / u_std

                    start_time_norm = (float(t_curr) - start_time_mean) / start_time_std
                    time_diff_norm = (float(time_step) - time_diff_mean) / time_diff_std

                    time_feats = torch.full(
                        (nc, 2), 0.0, device=device, dtype=torch.float32
                    )
                    time_feats[:, 0] = start_time_norm
                    time_feats[:, 1] = time_diff_norm

                    inputs = torch.cat(
                        [u_curr_norm, time_feats, mask.unsqueeze(-1)], dim=-1
                    ).unsqueeze(0)

                    pred = model(
                        latent_grid, coords_norm, inputs, coords_norm, enc, dec
                    )
                    pred_deriv = pred.squeeze(0) * der_std + der_mean
                    u_curr = u_curr + pred_deriv * time_step

                u_pred = u_curr

            diff = (u_pred - u_gt).abs()
            sq_diff = (u_pred - u_gt) ** 2

            rel_l1 = (diff.sum() / u_gt.abs().sum()).item()
            rel_l2 = (torch.sqrt(sq_diff.sum()) / torch.sqrt((u_gt**2).sum())).item()
            rmse = torch.sqrt(sq_diff.mean()).item()

            per_field_rel_l1 = []
            per_field_rmse = []
            for f in range(4):
                pf_rel = (diff[:, f].sum() / u_gt[:, f].abs().sum()).item()
                pf_rmse = torch.sqrt(sq_diff[:, f].mean()).item()
                per_field_rel_l1.append(pf_rel)
                per_field_rmse.append(pf_rmse)

            results[tau]["rel_l1"].append(rel_l1)
            results[tau]["rel_l2"].append(rel_l2)
            results[tau]["rmse"].append(rmse)
            results[tau]["per_field_rel_l1"].append(per_field_rel_l1)
            results[tau]["per_field_rmse"].append(per_field_rmse)

    aggregated = {}
    for tau in eval_tau_values:
        if len(results[tau]["rel_l1"]) > 0:
            aggregated[tau] = {
                "rel_l1": np.mean(results[tau]["rel_l1"]),
                "rel_l1_std": np.std(results[tau]["rel_l1"]),
                "rel_l2": np.mean(results[tau]["rel_l2"]),
                "rel_l2_std": np.std(results[tau]["rel_l2"]),
                "rmse": np.mean(results[tau]["rmse"]),
                "rmse_std": np.std(results[tau]["rmse"]),
                "per_field_rel_l1": np.mean(results[tau]["per_field_rel_l1"], axis=0),
                "per_field_rmse": np.mean(results[tau]["per_field_rmse"], axis=0),
                "n_samples": len(results[tau]["rel_l1"]),
            }

    return aggregated


def plot_rollout_results(rollout_results, save_path, time_step=3):
    taus = sorted(rollout_results.keys())

    rel_l1 = [rollout_results[t]["rel_l1"] for t in taus]
    rel_l2 = [rollout_results[t]["rel_l2"] for t in taus]
    rmse = [rollout_results[t]["rmse"] for t in taus]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(taus, rel_l1, "b-o", lw=2, ms=8)
    axes[0, 0].set_xlabel("τ (timesteps)")
    axes[0, 0].set_ylabel("Relative L1 Error")
    axes[0, 0].set_title("Relative L1 Error vs τ")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale("log")
    for t, v in zip(taus, rel_l1):
        axes[0, 0].annotate(
            f"{v:.3f}",
            (t, v),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    axes[0, 1].plot(taus, rel_l2, "g-s", lw=2, ms=8)
    axes[0, 1].set_xlabel("τ (timesteps)")
    axes[0, 1].set_ylabel("Relative L2 Error")
    axes[0, 1].set_title("Relative L2 Error vs τ (GraphViT metric)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale("log")
    for t, v in zip(taus, rel_l2):
        axes[0, 1].annotate(
            f"{v:.3f}",
            (t, v),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    axes[0, 2].plot(taus, rmse, "r-^", lw=2, ms=8)
    axes[0, 2].set_xlabel("τ (timesteps)")
    axes[0, 2].set_ylabel("RMSE")
    axes[0, 2].set_title("RMSE vs τ")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale("log")
    for t, v in zip(taus, rmse):
        axes[0, 2].annotate(
            f"{v:.3f}",
            (t, v),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    for f, name in enumerate(FIELD_NAMES):
        per_field = [rollout_results[t]["per_field_rel_l1"][f] for t in taus]
        axes[1, 0].plot(taus, per_field, "-o", lw=2, ms=6, label=name)
    axes[1, 0].set_xlabel("τ (timesteps)")
    axes[1, 0].set_ylabel("Relative L1 Error")
    axes[1, 0].set_title("Per-Field Relative L1 Error")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale("log")

    for f, name in enumerate(FIELD_NAMES):
        per_field = [rollout_results[t]["per_field_rmse"][f] for t in taus]
        axes[1, 1].plot(taus, per_field, "-s", lw=2, ms=6, label=name)
    axes[1, 1].set_xlabel("τ (timesteps)")
    axes[1, 1].set_ylabel("RMSE")
    axes[1, 1].set_title("Per-Field RMSE")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale("log")

    graphvit_taus = [1, 50, 250]
    graphvit_errors = [0.0811, 0.3495, 0.6357]

    our_comparison = []
    for gt in graphvit_taus:
        closest = min(taus, key=lambda x: abs(x - gt))
        if closest in rollout_results:
            our_comparison.append(rollout_results[closest]["rel_l2"])
        else:
            our_comparison.append(None)

    x_pos = np.arange(len(graphvit_taus))
    width = 0.35

    axes[1, 2].bar(
        x_pos - width / 2, graphvit_errors, width, label="GraphViT", color="steelblue"
    )
    valid_ours = [v if v is not None else 0 for v in our_comparison]
    axes[1, 2].bar(
        x_pos + width / 2, valid_ours, width, label="Ours", color="darkorange"
    )
    axes[1, 2].set_xlabel("τ (timesteps)")
    axes[1, 2].set_ylabel("Relative L2 Error")
    axes[1, 2].set_title("Comparison with GraphViT")
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels([f"+{t}" for t in graphvit_taus])
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log_info(f"Rollout results saved: {save_path}")


def plot_loss_curves(
    train_losses, val_losses, per_field_errors, save_path, eval_every=2
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    train_epochs = range(1, len(train_losses) + 1)
    val_epochs = range(eval_every, len(train_losses) + 1, eval_every)

    axes[0].plot(train_epochs, train_losses, label="Train", lw=2)
    axes[0].plot(val_epochs, val_losses, label="Val", lw=2, marker="o", ms=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].set_title("Loss (derivative MSE)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_epochs, val_losses, "b-", lw=2, marker="o", ms=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Loss")
    axes[1].set_yscale("log")
    axes[1].set_title("Validation Loss")
    axes[1].grid(True, alpha=0.3)

    if per_field_errors:
        for f, name in enumerate(FIELD_NAMES):
            errs = [per_field_errors[e][f] for e in sorted(per_field_errors.keys())]
            axes[2].plot(val_epochs, errs, label=name, lw=2, marker="o", ms=4)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Rel L1")
        axes[2].set_yscale("log")
        axes[2].legend()
        axes[2].set_title("Per-Field Validation Error")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log_info(f"Loss curves saved: {save_path}")


def plot_predictions(
    model,
    loader,
    latent_grid,
    encoder_nbrs,
    decoder_nbrs,
    device,
    stats,
    save_path,
    n_samples=4,
):
    model.eval()

    u_mean = stats["u_mean"].to(device)
    u_std = stats["u_std"].to(device)
    der_mean = stats["der_mean"].to(device)
    der_std = stats["der_std"].to(device)

    samples = []

    for i, (inputs, target, u_in_norm, coords, mask, dt, sample_idx) in enumerate(
        loader
    ):
        if i >= n_samples:
            break
        inputs = inputs.to(device)
        u_in_norm = u_in_norm.to(device)
        coords_dev = coords.to(device)
        dt = dt.to(device)

        enc = [
            {k: v.to(device) for k, v in s.items()} for s in encoder_nbrs[sample_idx]
        ]
        dec = [
            {k: v.to(device) for k, v in s.items()} for s in decoder_nbrs[sample_idx]
        ]

        with torch.no_grad():
            pred = model(latent_grid, coords_dev, inputs, coords_dev, enc, dec)

        pred_deriv = pred * der_std + der_mean
        pred_u_out = (u_in_norm * u_std + u_mean + pred_deriv * dt).cpu().numpy()[0]

        target_deriv = target * der_std.cpu() + der_mean.cpu()
        tgt_u_out = (
            u_in_norm.cpu() * u_std.cpu() + u_mean.cpu() + target_deriv * dt.cpu()
        ).numpy()[0]

        samples.append((coords.numpy(), tgt_u_out, pred_u_out, mask.numpy()))

    fig, axes = plt.subplots(n_samples, 12, figsize=(28, n_samples * 2.5))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, (coords_np, tgt, pred, mask_np) in enumerate(samples):
        err = np.abs(pred - tgt)
        for f in range(4):
            vmin = min(tgt[:, f].min(), pred[:, f].min())
            vmax = max(tgt[:, f].max(), pred[:, f].max())
            for col, (data, title) in enumerate(
                [
                    (tgt[:, f], f"{FIELD_NAMES[f]} GT"),
                    (pred[:, f], f"{FIELD_NAMES[f]} Pred"),
                    (err[:, f], f"{FIELD_NAMES[f]} Err"),
                ]
            ):
                ax = axes[row, f * 3 + col]
                cmap = "hot" if col == 2 else "RdBu_r"
                vm = (0, np.percentile(err[:, f], 99)) if col == 2 else (vmin, vmax)
                sc = ax.scatter(
                    coords_np[:, 0],
                    coords_np[:, 1],
                    c=data,
                    s=0.5,
                    cmap=cmap,
                    vmin=vm[0],
                    vmax=vm[1],
                )
                ax.scatter(
                    coords_np[mask_np > 0.5, 0],
                    coords_np[mask_np > 0.5, 1],
                    c="lime",
                    s=2,
                    alpha=0.7,
                )
                ax.set_title(title, fontsize=9)
                ax.set_aspect("equal")
                ax.axis("off")
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    log_info(f"Predictions saved: {save_path}")


def save_config(config, model_config, train_config, save_path):
    config_dict = {
        "model": {
            "latent_tokens_size": list(config.latent_tokens_size),
            "magno": {
                "coord_dim": model_config.args.magno.coord_dim,
                "radius": model_config.args.magno.radius,
                "hidden_size": model_config.args.magno.hidden_size,
                "mlp_layers": model_config.args.magno.mlp_layers,
                "lifting_channels": model_config.args.magno.lifting_channels,
                "scales": list(model_config.args.magno.scales),
            },
            "transformer": {
                "patch_size": model_config.args.transformer.patch_size,
                "hidden_size": model_config.args.transformer.hidden_size,
                "num_layers": model_config.args.transformer.num_layers,
            },
        },
        "training": {
            "train_size": train_config.train_size,
            "val_size": train_config.val_size,
            "test_size": train_config.test_size,
            "max_time_diff": train_config.max_time_diff,
            "time_step": train_config.time_step,
            "pairs_per_sample": train_config.pairs_per_sample,
            "n_epochs": train_config.n_epochs,
            "lr": train_config.lr,
            "max_lr": train_config.max_lr,
            "min_lr": train_config.min_lr,
            "final_lr": train_config.final_lr,
            "weight_decay": train_config.weight_decay,
            "accumulation_steps": train_config.accumulation_steps,
            "seed": train_config.seed,
        },
        "evaluation": {
            "eval_tau_values": train_config.eval_tau_values,
            "direct_tau_max": train_config.direct_tau_max,
        },
    }

    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    log_info(f"Config saved: {save_path}")


def save_results_json(results, rollout_results, save_path):
    results_dict = {
        "single_step": {
            "train_loss": results["train_loss"],
            "train_rel_l1": results["train_rel"],
            "val_loss": results["val_loss"],
            "val_rel_l1": results["val_rel"],
            "test_loss": results["test_loss"],
            "test_rel_l1": results["test_rel"],
            "best_epoch": results["best_epoch"],
            "n_params": results["n_params"],
        },
        "rollout": {},
    }

    for tau, r in rollout_results.items():
        results_dict["rollout"][str(tau)] = {
            "rel_l1": r["rel_l1"],
            "rel_l1_std": r.get("rel_l1_std", 0),
            "rel_l2": r["rel_l2"],
            "rel_l2_std": r.get("rel_l2_std", 0),
            "rmse": r["rmse"],
            "rmse_std": r.get("rmse_std", 0),
            "per_field_rel_l1": r["per_field_rel_l1"].tolist(),
            "per_field_rmse": r["per_field_rmse"].tolist(),
            "n_samples": r.get("n_samples", 0),
        }

    graphvit_comparison = {
        "1": {
            "graphvit": 0.0811,
            "ours": results_dict["rollout"].get("3", {}).get("rel_l2", None),
        },
        "50": {
            "graphvit": 0.3495,
            "ours": results_dict["rollout"]
            .get("51", results_dict["rollout"].get("48", {}))
            .get("rel_l2", None),
        },
        "250": {
            "graphvit": 0.6357,
            "ours": results_dict["rollout"]
            .get("249", results_dict["rollout"].get("252", {}))
            .get("rel_l2", None),
        },
    }
    results_dict["graphvit_comparison"] = graphvit_comparison

    with open(save_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    log_info(f"Results JSON saved: {save_path}")


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    log_info(f"World size: {world_size}, Rank: {rank}, Device: {device}")
    log_info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    vtkhdf_path = "/var/llm/sci-data/INS_2D_EAGLE-Drone-Wake.vtkhdf"
    neighbors_path = "/var/llm/sci-data/eagle_neighbors.pt"

    model_config = build_config()
    train_config = build_training_config()

    log_info("=" * 70)
    log_info("CONFIGURATION")
    log_info("=" * 70)
    log_info(
        f"Training: epochs={train_config.n_epochs}, lr={train_config.lr}, "
        f"max_lr={train_config.max_lr}, weight_decay={train_config.weight_decay}"
    )
    log_info(
        f"Data: train={train_config.train_size}, val={train_config.val_size}, "
        f"test={train_config.test_size}"
    )
    log_info(
        f"Time: max_diff={train_config.max_time_diff}, step={train_config.time_step}, "
        f"pairs_per_sample={train_config.pairs_per_sample}"
    )
    log_info(
        f"Batch: accumulation_steps={train_config.accumulation_steps} "
        f"(effective batch={train_config.accumulation_steps * world_size})"
    )
    log_info(f"Eval taus: {train_config.eval_tau_values}")
    log_info("=" * 70)

    with h5py.File(vtkhdf_path, "r") as f:
        n_samples = len(f["SampleNodeCount"])

    rng = np.random.default_rng(train_config.seed)
    perm = rng.permutation(n_samples)
    train_idx = perm[: train_config.train_size].tolist()
    val_idx = perm[
        train_config.train_size : train_config.train_size + train_config.val_size
    ].tolist()
    test_idx = perm[-train_config.test_size :].tolist()

    stats = compute_global_stats(
        vtkhdf_path,
        train_idx,
        max_time_diff=train_config.max_time_diff,
        time_step=train_config.time_step,
    )

    log_info(f"u_mean:   {stats['u_mean']}")
    log_info(f"u_std:    {stats['u_std']}")
    log_info(f"der_mean: {stats['der_mean']}")
    log_info(f"der_std:  {stats['der_std']}")
    log_info(
        f"start_time: mean={stats['start_time_mean']:.2f}, std={stats['start_time_std']:.2f}"
    )
    log_info(
        f"time_diff:  mean={stats['time_diff_mean']:.2f}, std={stats['time_diff_std']:.2f}"
    )

    max_rollout_tau = max(train_config.eval_tau_values)
    all_indices = train_idx + val_idx + test_idx
    shared_data = load_shared_data(
        vtkhdf_path,
        neighbors_path,
        all_indices,
        train_config.max_time_diff,
        train_config.time_step,
        n_timesteps=990,
    )

    n_timesteps = shared_data["n_timesteps"]

    (
        train_loader,
        val_loader,
        test_loader,
        latent_grid,
        encoder_nbrs,
        decoder_nbrs,
        train_sampler,
    ) = create_dataloaders(
        shared_data,
        stats,
        train_idx,
        val_idx,
        test_idx,
        max_time_diff=train_config.max_time_diff,
        time_step=train_config.time_step,
        pairs_per_sample=train_config.pairs_per_sample,
        n_timesteps=n_timesteps,
        world_size=world_size,
        rank=rank,
    )

    latent_grid = latent_grid.to(device)
    log_info(
        f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )

    model = GAOT(input_size=7, output_size=4, config=model_config).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info(f"Model parameters: {n_params / 1e6:.2f}M")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )

    scheduler = MixScheduler(
        optimizer=optimizer,
        total_epochs=train_config.n_epochs,
        lr=train_config.lr,
        max_lr=train_config.max_lr,
        min_lr=train_config.min_lr,
        final_lr=train_config.final_lr,
        warmup_fraction=train_config.warmup_fraction,
        cosine_fraction=train_config.cosine_fraction,
    )

    ckpt_dir = Path(".ckpt/eagle")
    results_dir = Path(".results/eagle")
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        save_config(
            model_config, model_config, train_config, results_dir / "config.json"
        )

    if world_size > 1:
        dist.barrier()

    best_val, best_epoch, best_state = float("inf"), 0, None
    train_losses, val_losses, per_field_errors = [], [], {}

    log_info("=" * 70)
    log_info("TRAINING")
    log_info("=" * 70)

    for epoch in range(train_config.n_epochs):
        lr = scheduler.get_lr()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            latent_grid,
            encoder_nbrs,
            decoder_nbrs,
            device,
            world_size=world_size,
            sampler=train_sampler,
            epoch=epoch,
            accumulation_steps=train_config.accumulation_steps,
        )
        train_losses.append(train_loss)

        scheduler.step()

        if (epoch + 1) % train_config.eval_every == 0:
            val_loss, val_rel, val_pf = evaluate(
                model,
                val_loader,
                criterion,
                latent_grid,
                encoder_nbrs,
                decoder_nbrs,
                device,
                stats,
                world_size=world_size,
            )
            val_losses.append(val_loss)
            per_field_errors[epoch] = val_pf

            pf_str = " ".join(
                [f"{FIELD_NAMES[f]}:{val_pf[f] * 100:.1f}%" for f in range(4)]
            )
            log_info(
                f"Epoch {epoch + 1:3d} | lr={lr:.2e} | train={train_loss:.4f} | "
                f"val={val_loss:.4f} | rel={val_rel * 100:.1f}% | {pf_str}"
            )

            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                if is_main_process():
                    state_dict = (
                        model.module.state_dict()
                        if world_size > 1
                        else model.state_dict()
                    )
                    best_state = deepcopy(state_dict)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": best_state,
                            "stats": stats,
                            "config": {
                                "model": model_config,
                                "train": train_config,
                            },
                        },
                        ckpt_dir / "best.pt",
                    )
                    log_info("  -> saved best checkpoint")
        else:
            log_info(f"Epoch {epoch + 1:3d} | lr={lr:.2e} | train={train_loss:.4f}")

    log_info(f"Training complete. Best epoch: {best_epoch + 1}")

    if is_main_process() and best_state is not None:
        if world_size > 1:
            model.module.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)

        eval_model = model.module if world_size > 1 else model

        log_info("=" * 70)
        log_info("FINAL EVALUATION")
        log_info("=" * 70)

        train_loss, train_rel, train_pf = evaluate(
            eval_model,
            train_loader,
            criterion,
            latent_grid,
            encoder_nbrs,
            decoder_nbrs,
            device,
            stats,
            world_size=1,
        )
        val_loss, val_rel, val_pf = evaluate(
            eval_model,
            val_loader,
            criterion,
            latent_grid,
            encoder_nbrs,
            decoder_nbrs,
            device,
            stats,
            world_size=1,
        )
        test_loss, test_rel, test_pf = evaluate(
            eval_model,
            test_loader,
            criterion,
            latent_grid,
            encoder_nbrs,
            decoder_nbrs,
            device,
            stats,
            world_size=1,
        )

        log_info("SINGLE-STEP RESULTS (Best epoch {})".format(best_epoch + 1))
        log_info("-" * 70)
        for name, loss, rel, pf in [
            ("Train", train_loss, train_rel, train_pf),
            ("Val", val_loss, val_rel, val_pf),
            ("Test", test_loss, test_rel, test_pf),
        ]:
            log_info(
                f"{name:<6} Loss={loss:.4f} Rel={rel * 100:.1f}% "
                + " ".join([f"{FIELD_NAMES[f]}:{pf[f] * 100:.1f}%" for f in range(4)])
            )

        log_info("=" * 70)
        log_info("AUTOREGRESSIVE ROLLOUT EVALUATION")
        log_info("=" * 70)

        rollout_results = autoregressive_rollout(
            eval_model,
            shared_data,
            test_idx,
            stats,
            latent_grid,
            encoder_nbrs,
            decoder_nbrs,
            device,
            time_step=train_config.time_step,
            eval_tau_values=train_config.eval_tau_values,
            direct_tau_max=train_config.direct_tau_max,
        )

        log_info(
            f"{'τ':<8} {'Method':<12} {'Rel L1':<10} {'Rel L2':<10} {'RMSE':<10} "
            f"{'VX':<8} {'VY':<8} {'PS':<8} {'PG':<8}"
        )
        log_info("-" * 90)

        for tau in sorted(rollout_results.keys()):
            r = rollout_results[tau]
            method = "direct" if tau <= train_config.direct_tau_max else "AR"
            pf = r["per_field_rel_l1"]
            log_info(
                f"{tau:<8} {method:<12} {r['rel_l1']:.4f}     {r['rel_l2']:.4f}     "
                f"{r['rmse']:.4f}     {pf[0]:.4f}   {pf[1]:.4f}   {pf[2]:.4f}   {pf[3]:.4f}"
            )

        log_info("=" * 70)
        log_info("COMPARISON WITH GRAPHVIT (EAGLE PAPER)")
        log_info("=" * 70)
        log_info("GraphViT results: +1=0.0811, +50=0.3495, +250=0.6357")

        comparison_taus = [(3, 1), (51, 50), (249, 250)]
        for our_tau, paper_tau in comparison_taus:
            closest = min(rollout_results.keys(), key=lambda x: abs(x - our_tau))
            if closest in rollout_results:
                our_val = rollout_results[closest]["rel_l2"]
                paper_val = {1: 0.0811, 50: 0.3495, 250: 0.6357}.get(paper_tau, None)
                if paper_val:
                    diff = our_val - paper_val
                    pct = (diff / paper_val) * 100
                    status = "BETTER" if diff < 0 else "WORSE"
                    log_info(
                        f"τ≈{paper_tau}: Ours={our_val:.4f} vs GraphViT={paper_val:.4f} "
                        f"({status}, {pct:+.1f}%)"
                    )

        results = {
            "train_loss": train_loss,
            "train_rel": train_rel,
            "train_per_field": train_pf,
            "val_loss": val_loss,
            "val_rel": val_rel,
            "val_per_field": val_pf,
            "test_loss": test_loss,
            "test_rel": test_rel,
            "test_per_field": test_pf,
            "best_epoch": best_epoch,
            "n_params": sum(p.numel() for p in model.parameters()),
        }

        plot_loss_curves(
            train_losses,
            val_losses,
            per_field_errors,
            results_dir / "loss_curves.png",
            train_config.eval_every,
        )
        plot_predictions(
            eval_model,
            test_loader,
            latent_grid,
            encoder_nbrs,
            decoder_nbrs,
            device,
            stats,
            results_dir / "test_predictions.png",
            n_samples=4,
        )
        plot_rollout_results(
            rollout_results,
            results_dir / "rollout_results.png",
            time_step=train_config.time_step,
        )

        save_results_json(results, rollout_results, results_dir / "results.json")

        np.savez(
            results_dir / "results.npz",
            train_losses=train_losses,
            val_losses=val_losses,
            **{k: v for k, v in results.items()},
            rollout_taus=list(rollout_results.keys()),
            rollout_rel_l1=[
                rollout_results[t]["rel_l1"] for t in sorted(rollout_results.keys())
            ],
            rollout_rel_l2=[
                rollout_results[t]["rel_l2"] for t in sorted(rollout_results.keys())
            ],
            rollout_rmse=[
                rollout_results[t]["rmse"] for t in sorted(rollout_results.keys())
            ],
        )

        log_info("=" * 70)
        log_info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_info(f"Results saved to: {results_dir}")
        log_info("=" * 70)

    cleanup_distributed()


if __name__ == "__main__":
    main()
