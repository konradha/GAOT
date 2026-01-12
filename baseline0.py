import numpy as np
import torch
import torch.nn as nn
import h5py
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
import logging

from src.model.gaot import GAOT
from src.model.layers.magno import MAGNOConfig
from src.model.layers.attn import TransformerConfig

VTKHDF_PATH = Path("/var/llm/sci-data/INS_2D_EAGLE-Drone-Wake.vtkhdf")
NEIGHBORS_PATH = Path("/var/llm/sci-data/eagle_neighbors.pt")
CKPT_DIR = Path(".ckpt/baseline0")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SIZE = 100
VAL_SIZE = 50
TEST_SIZE = 50
N_TIMESTEPS = 990
TIME_PAIRS_PER_FORWARD = 8
ITERS_PER_EPOCH = 2000

N_EPOCHS = 100
LR = 8e-4
MAX_LR = 1e-3
MIN_LR = 1e-4
FINAL_LR = 5e-5
WEIGHT_DECAY = 1e-5
EVAL_EVERY = 2
SEED = 1

EVAL_TAUS = [1, 50, 250]
ALPHA = 0.1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MixScheduler:
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


def weighted_rms_loss(pred, target):
    vel_rms = torch.sqrt(torch.mean((pred[..., :2] - target[..., :2]) ** 2))
    pres_rms = torch.sqrt(torch.mean((pred[..., 2:] - target[..., 2:]) ** 2))
    return vel_rms + ALPHA * pres_rms


def build_config():
    class Config:
        pass

    config = Config()
    config.latent_tokens_size = [64, 64]
    config.args = Config()
    config.args.magno = MAGNOConfig(
        coord_dim=2,
        radius=0.05,
        hidden_size=64,
        mlp_layers=3,
        lifting_channels=64,
        scales=[1.0, 2.0],
        use_attention=True,
        use_geoembed=True,
        embedding_method="statistical",
        transform_type="linear",
        precompute_edges=True,
    )
    config.args.transformer = TransformerConfig(
        patch_size=2,
        hidden_size=256,
        num_layers=3,
        positional_embedding="absolute",
    )
    return config


def compute_stats(vtkhdf_path, sample_indices, n_samples_for_stats=50):
    samples_to_use = sample_indices[: min(n_samples_for_stats, len(sample_indices))]

    with h5py.File(vtkhdf_path, "r") as f:
        node_counts = f["SampleNodeCount"][:]
        vx, vy = f["VTKHDF/PointData/VX"], f["VTKHDF/PointData/VY"]
        ps, pg = f["VTKHDF/PointData/PS"], f["VTKHDF/PointData/PG"]

        u_sums = np.zeros(4, dtype=np.float64)
        u_sq_sums = np.zeros(4, dtype=np.float64)
        u_count = 0

        res_sums = np.zeros(4, dtype=np.float64)
        res_sq_sums = np.zeros(4, dtype=np.float64)
        res_count = 0

        for sample_idx in samples_to_use:
            nc = int(node_counts[sample_idx])
            base = sum(int(node_counts[s]) * N_TIMESTEPS for s in range(sample_idx))

            prev_vals = None
            for t in range(N_TIMESTEPS):
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
                    residual = vals - prev_vals
                    res_sums += residual.sum(axis=0)
                    res_sq_sums += (residual**2).sum(axis=0)
                    res_count += nc

                prev_vals = vals

    u_mean = u_sums / u_count
    u_std = np.maximum(np.sqrt(u_sq_sums / u_count - u_mean**2), 1e-6)
    res_mean = res_sums / res_count
    res_std = np.maximum(np.sqrt(res_sq_sums / res_count - res_mean**2), 1e-6)

    return {
        "u_mean": torch.tensor(u_mean, dtype=torch.float32),
        "u_std": torch.tensor(u_std, dtype=torch.float32),
        "res_mean": torch.tensor(res_mean, dtype=torch.float32),
        "res_std": torch.tensor(res_std, dtype=torch.float32),
    }


def load_data(vtkhdf_path, neighbors_path, all_indices, device):
    logging.info("Loading VTKHDF data...")
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

    for sample_idx in all_indices:
        nc = int(node_counts[sample_idx])
        ns = int(node_offsets[sample_idx])
        base = sum(int(node_counts[s]) * N_TIMESTEPS for s in range(sample_idx))
        coords[sample_idx] = points[ns : ns + nc].copy()

        for t in range(N_TIMESTEPS):
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
    logging.info("Loading neighbors and caching on GPU...")

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

    coords_norm = {}
    for sample_idx in all_indices:
        c = coords[sample_idx]
        c_norm = 2.0 * (c - coord_min) / (coord_max - coord_min) - 1.0
        coords_norm[sample_idx] = torch.from_numpy(c_norm.astype(np.float32)).to(device)

    logging.info("Data loading complete.")

    return {
        "fields": fields,
        "masks": masks,
        "coords": coords,
        "coords_norm": coords_norm,
        "latent_grid": latent_grid,
        "coord_min": coord_min,
        "coord_max": coord_max,
        "encoder_nbrs": encoder_nbrs,
        "decoder_nbrs": decoder_nbrs,
    }


def train_epoch(model, data, train_idx, stats, optimizer, device):
    model.train()
    total_loss = 0.0

    u_mean = stats["u_mean"].to(device)
    u_std = stats["u_std"].to(device)
    res_mean = stats["res_mean"].to(device)
    res_std = stats["res_std"].to(device)

    latent_grid = data["latent_grid"]
    encoder_nbrs = data["encoder_nbrs"]
    decoder_nbrs = data["decoder_nbrs"]
    coords_norm = data["coords_norm"]
    fields = data["fields"]
    masks = data["masks"]

    pbar = tqdm(range(ITERS_PER_EPOCH), desc="Train", leave=False)

    for _ in pbar:
        sample_idx = train_idx[np.random.randint(len(train_idx))]

        t_ins = np.random.randint(0, N_TIMESTEPS - 1, size=TIME_PAIRS_PER_FORWARD)

        u_in_list, u_out_list, mask_list = [], [], []
        for t_in in t_ins:
            u_in_list.append(fields[(sample_idx, t_in)])
            u_out_list.append(fields[(sample_idx, t_in + 1)])
            mask_list.append(masks[(sample_idx, t_in)])

        u_in = torch.from_numpy(np.stack(u_in_list)).to(device)
        u_out = torch.from_numpy(np.stack(u_out_list)).to(device)
        mask = torch.from_numpy(np.stack(mask_list)).to(device)

        u_in_norm = (u_in - u_mean) / u_std
        residual = u_out - u_in
        target = (residual - res_mean) / res_std

        inputs = torch.cat([u_in_norm, mask.unsqueeze(-1)], dim=-1)

        coords_n = coords_norm[sample_idx]
        enc = encoder_nbrs[sample_idx]
        dec = decoder_nbrs[sample_idx]

        optimizer.zero_grad()
        pred = model(latent_grid, coords_n, inputs, coords_n, enc, dec)
        loss = weighted_rms_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / ITERS_PER_EPOCH


@torch.no_grad()
def evaluate(model, data, sample_indices, stats, device, max_iters=500):
    model.eval()
    total_loss = 0.0
    total_rel_l2 = 0.0
    n = 0

    u_mean = stats["u_mean"].to(device)
    u_std = stats["u_std"].to(device)
    res_mean = stats["res_mean"].to(device)
    res_std = stats["res_std"].to(device)

    latent_grid = data["latent_grid"]
    encoder_nbrs = data["encoder_nbrs"]
    decoder_nbrs = data["decoder_nbrs"]
    coords_norm = data["coords_norm"]
    fields = data["fields"]
    masks = data["masks"]

    iters = min(max_iters, len(sample_indices) * 10)
    pbar = tqdm(range(iters), desc="Eval", leave=False)

    for _ in pbar:
        sample_idx = sample_indices[np.random.randint(len(sample_indices))]
        t_in = np.random.randint(0, N_TIMESTEPS - 1)

        u_in = torch.from_numpy(fields[(sample_idx, t_in)]).unsqueeze(0).to(device)
        u_out = torch.from_numpy(fields[(sample_idx, t_in + 1)]).unsqueeze(0).to(device)
        mask = torch.from_numpy(masks[(sample_idx, t_in)]).unsqueeze(0).to(device)

        u_in_norm = (u_in - u_mean) / u_std
        residual = u_out - u_in
        target = (residual - res_mean) / res_std

        inputs = torch.cat([u_in_norm, mask.unsqueeze(-1)], dim=-1)

        coords_n = coords_norm[sample_idx]
        enc = encoder_nbrs[sample_idx]
        dec = decoder_nbrs[sample_idx]

        pred = model(latent_grid, coords_n, inputs, coords_n, enc, dec)
        loss = weighted_rms_loss(pred, target)
        total_loss += loss.item()

        pred_res = pred.squeeze(0) * res_std + res_mean
        u_pred = u_in.squeeze(0) + pred_res

        target_res = target.squeeze(0) * res_std + res_mean
        u_gt = u_in.squeeze(0) + target_res

        rel_l2 = torch.sqrt(((u_pred - u_gt) ** 2).sum()) / torch.sqrt((u_gt**2).sum())
        total_rel_l2 += rel_l2.item()
        n += 1

    return total_loss / n, total_rel_l2 / n


@torch.no_grad()
def ar_rollout(model, data, sample_indices, stats, device):
    model.eval()

    u_mean = stats["u_mean"].to(device)
    u_std = stats["u_std"].to(device)
    res_mean = stats["res_mean"].to(device)
    res_std = stats["res_std"].to(device)

    latent_grid = data["latent_grid"]
    encoder_nbrs = data["encoder_nbrs"]
    decoder_nbrs = data["decoder_nbrs"]
    coords_norm = data["coords_norm"]
    fields = data["fields"]
    masks = data["masks"]

    results = {tau: [] for tau in EVAL_TAUS}

    for sample_idx in tqdm(sample_indices, desc="AR Rollout", leave=False):
        coords_n = coords_norm[sample_idx]
        enc = encoder_nbrs[sample_idx]
        dec = decoder_nbrs[sample_idx]

        u_t0 = torch.from_numpy(fields[(sample_idx, 0)]).to(device)

        for tau in EVAL_TAUS:
            if tau > N_TIMESTEPS - 1:
                continue

            u_gt = torch.from_numpy(fields[(sample_idx, tau)]).to(device)
            u_curr = u_t0.clone()

            for step in range(tau):
                mask = torch.from_numpy(masks[(sample_idx, step)]).to(device)
                u_curr_norm = (u_curr - u_mean) / u_std
                inputs = torch.cat([u_curr_norm, mask.unsqueeze(-1)], dim=-1).unsqueeze(
                    0
                )

                pred = model(latent_grid, coords_n, inputs, coords_n, enc, dec)
                pred_res = pred.squeeze(0) * res_std + res_mean
                u_curr = u_curr + pred_res

            rel_l2 = (
                torch.sqrt(((u_curr - u_gt) ** 2).sum()) / torch.sqrt((u_gt**2).sum())
            ).item()
            results[tau].append(rel_l2)

    return {
        tau: {"mean": np.mean(vals), "std": np.std(vals)}
        for tau, vals in results.items()
        if vals
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Started: {datetime.now()}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with h5py.File(VTKHDF_PATH, "r") as f:
        n_samples = len(f["SampleNodeCount"])

    perm = np.random.permutation(n_samples)
    train_idx = perm[:TRAIN_SIZE].tolist()
    val_idx = perm[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE].tolist()
    test_idx = perm[-TEST_SIZE:].tolist()

    logging.info(
        f"Samples: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    stats = compute_stats(VTKHDF_PATH, train_idx)
    logging.info(f"u_mean: {stats['u_mean']}")
    logging.info(f"u_std: {stats['u_std']}")
    logging.info(f"res_mean: {stats['res_mean']}")
    logging.info(f"res_std: {stats['res_std']}")

    all_indices = train_idx + val_idx + test_idx
    data = load_data(VTKHDF_PATH, NEIGHBORS_PATH, all_indices, device)

    config = build_config()
    model = GAOT(input_size=5, output_size=4, config=config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = MixScheduler(optimizer, N_EPOCHS, LR, MAX_LR, MIN_LR, FINAL_LR)

    best_val_loss = float("inf")
    best_state = None

    logging.info("=" * 60)
    logging.info("TRAINING")
    logging.info("=" * 60)

    for epoch in range(N_EPOCHS):
        lr = scheduler.get_lr()
        train_loss = train_epoch(model, data, train_idx, stats, optimizer, device)
        scheduler.step()

        if (epoch + 1) % EVAL_EVERY == 0:
            val_loss, val_rel = evaluate(model, data, val_idx, stats, device)
            logging.info(
                f"Epoch {epoch + 1:3d} | lr={lr:.2e} | train={train_loss:.4f} | val={val_loss:.4f} | rel_l2={val_rel * 100:.2f}%"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                torch.save(
                    {"epoch": epoch, "model": best_state, "stats": stats},
                    CKPT_DIR / "best.pt",
                )
                logging.info("  -> saved best")
        else:
            logging.info(f"Epoch {epoch + 1:3d} | lr={lr:.2e} | train={train_loss:.4f}")

    logging.info("=" * 60)
    logging.info("FINAL EVALUATION")
    logging.info("=" * 60)

    model.load_state_dict(best_state)

    test_loss, test_rel = evaluate(model, data, test_idx, stats, device)
    logging.info(f"Test: loss={test_loss:.4f}, rel_l2={test_rel * 100:.2f}%")

    logging.info("AR Rollout on test set:")
    rollout = ar_rollout(model, data, test_idx, stats, device)

    logging.info(f"{'tau':<8} {'Rel L2':<12} {'Std':<12}")
    logging.info("-" * 32)
    for tau in EVAL_TAUS:
        if tau in rollout:
            r = rollout[tau]
            logging.info(f"{tau:<8} {r['mean'] * 100:.2f}%       {r['std'] * 100:.2f}%")

    logging.info("GraphViT targets: +1=8.11%, +50=34.95%, +250=63.57%")
    logging.info(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
