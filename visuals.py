import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.dataset import DATASET_METADATA
from src.model.gaot import GAOT
from src.core.default_configs import ModelConfig, DatasetConfig, merge_config
from omegaconf import OmegaConf

plt.style.use("dark_background")


def get_sample_path(data_root, sample_idx):
    tri_root = Path(data_root) / "Tri"
    scene_dirs = sorted(
        [d for d in tri_root.iterdir() if d.is_dir()], key=lambda x: int(x.name)
    )

    flat_idx = 0
    for scene_dir in scene_dirs:
        sim_dirs = sorted(
            [d for d in scene_dir.iterdir() if d.is_dir()], key=lambda x: int(x.name)
        )
        for sim_dir in sim_dirs:
            if flat_idx == sample_idx:
                return sim_dir
            flat_idx += 1
    raise ValueError(f"Sample {sample_idx} not found")


def load_triangles(data_root, sample_idx, t):
    sample_path = get_sample_path(data_root, sample_idx)
    tri = np.load(sample_path / "triangles.npy")
    return tri[t]


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model_config = merge_config(ModelConfig, config.model)
    metadata = DATASET_METADATA[config.dataset.metaname]

    model_config.args.magno.coord_dim = 2
    n_vars = len(metadata.active_variables)

    model = GAOT(input_size=n_vars + 2, output_size=n_vars, config=model_config)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    latent_queries = torch.stack(
        torch.meshgrid(
            torch.linspace(
                metadata.domain_x[0][0],
                metadata.domain_x[1][0],
                model_config.latent_tokens_size[0],
            ),
            torch.linspace(
                metadata.domain_x[0][1],
                metadata.domain_x[1][1],
                model_config.latent_tokens_size[1],
            ),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)

    return model, latent_queries, metadata


def load_data(nc_path, active_vars, n_samples=20, n_timesteps=7):
    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        u = ds["u"].values[..., active_vars]
        x = ds["x"].values
        mask = ds["mask"].values
    x_subset = x[:n_samples, :n_timesteps]
    u_subset = u[:n_samples, :n_timesteps]

    valid = np.abs(x_subset).sum(axis=-1) > 1e-6
    valid_expanded = valid[..., np.newaxis]

    u_masked = np.where(valid_expanded, u_subset, np.nan)
    u_mean = np.nanmean(u_masked.reshape(-1, u_subset.shape[-1]), axis=0)
    u_std = np.nanstd(u_masked.reshape(-1, u_subset.shape[-1]), axis=0) + 1e-10

    return u, x, mask, u_mean, u_std


def predict(model, u_in, x_coord, latent_q, u_mean, u_std, device):
    u_norm = (u_in - u_mean) / u_std
    n = u_in.shape[0]

    inp = (
        torch.cat(
            [
                torch.from_numpy(u_norm).float(),
                torch.zeros(n, 1),
                torch.ones(n, 1) * 0.1,
            ],
            dim=-1,
        )
        .unsqueeze(0)
        .to(device)
    )

    x_t = torch.from_numpy(x_coord).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(
            latent_tokens_coord=latent_q.to(device),
            xcoord=x_t,
            pndata=inp,
            encoder_nbrs=None,
            decoder_nbrs=None,
        )

    return out.squeeze(0).cpu().numpy() * u_std + u_mean


def plot_comparison(
    sample_idx,
    t_start,
    t_end,
    u,
    x,
    mask,
    triangles,
    model,
    latent_q,
    u_mean,
    u_std,
    device,
    save_dir,
):
    u_in = u[sample_idx, t_start]
    u_target = u[sample_idx, t_end]
    x_coord = x[sample_idx, t_start]

    pred = predict(model, u_in, x_coord, latent_q, u_mean, u_std, device)
    error = np.abs(u_target - pred)

    tri = Triangulation(x_coord[:, 0], x_coord[:, 1], triangles)

    var_names = ["VX", "VY", "PS", "PG"]
    n_vars = u_in.shape[-1]

    fig, axes = plt.subplots(n_vars, 4, figsize=(20, 4 * n_vars), facecolor="#0d1117")

    for v in range(n_vars):
        vmin = min(u_in[:, v].min(), u_target[:, v].min(), pred[:, v].min())
        vmax = max(u_in[:, v].max(), u_target[:, v].max(), pred[:, v].max())
        cmap = "RdBu_r" if v < 2 else "viridis"

        for ax in axes[v]:
            ax.set_facecolor("#0d1117")
            ax.set_aspect("equal")
            ax.axis("off")

        im0 = axes[v, 0].tripcolor(
            tri, u_in[:, v], cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
        )
        axes[v, 0].set_title(f"{var_names[v]} @ t={t_start} (input)", color="white")
        plt.colorbar(im0, ax=axes[v, 0], shrink=0.7)

        im1 = axes[v, 1].tripcolor(
            tri, u_target[:, v], cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
        )
        axes[v, 1].set_title(f"{var_names[v]} @ t={t_end} (target)", color="white")
        plt.colorbar(im1, ax=axes[v, 1], shrink=0.7)

        im2 = axes[v, 2].tripcolor(
            tri, pred[:, v], cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
        )
        axes[v, 2].set_title(f"{var_names[v]} @ t={t_end} (pred)", color="white")
        plt.colorbar(im2, ax=axes[v, 2], shrink=0.7)

        im3 = axes[v, 3].tripcolor(tri, error[:, v], cmap="hot", shading="flat")
        axes[v, 3].set_title(f"{var_names[v]} |error|", color="white")
        plt.colorbar(im3, ax=axes[v, 3], shrink=0.7)

    fig.suptitle(
        f"Sample {sample_idx}: t={t_start} → t={t_end}", color="white", fontsize=16
    )
    plt.tight_layout()

    save_path = Path(save_dir) / f"pred_sample{sample_idx}_t{t_start}_to_t{t_end}.png"
    fig.savefig(save_path, dpi=150, facecolor="#0d1117", bbox_inches="tight")
    plt.close()

    rel_err = np.sqrt((error**2).mean(axis=0)) / (np.abs(u_target).mean(axis=0) + 1e-10)
    return rel_err


def main():
    config_path = "config/examples/time_dep/eagle_tri.json"
    ckpt_path = ".ckpt/eagle/eagle_tri.pt"
    nc_path = "/var/llm/sci-data/eagle_tri_aggregated.nc"
    data_root = "/var/llm/sci-data/Eagle_dataset"
    save_dir = ".results/eagle/predictions"

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, latent_q, metadata = load_model(config_path, ckpt_path, device)
    u, x, mask, u_mean, u_std = load_data(nc_path, metadata.active_variables)

    samples = [0, 5, 10, 15, 19]
    time_pairs = [(0, 2), (0, 4), (0, 6)]

    results = []
    tasks = [(s, t0, t1) for s in samples for t0, t1 in time_pairs]

    for sample_idx, t_start, t_end in tqdm(tasks, desc="Generating"):
        triangles = load_triangles(data_root, sample_idx, t_start)
        rel_err = plot_comparison(
            sample_idx,
            t_start,
            t_end,
            u,
            x,
            mask,
            triangles,
            model,
            latent_q,
            u_mean,
            u_std,
            device,
            save_dir,
        )
        results.append(
            {"sample": sample_idx, "t0": t_start, "t1": t_end, "rel_err": rel_err}
        )

    print(f"\nSaved to {save_dir}")
    mean_err = np.mean([r["rel_err"] for r in results], axis=0)
    for i, name in enumerate(["VX", "VY", "PS", "PG"][: len(mean_err)]):
        print(f"  {name}: {mean_err[i]:.4f}")


if __name__ == "__main__":
    main()
