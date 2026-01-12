import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.tri import Triangulation
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
L = logging.info

DATA_ROOT = Path("/var/llm/sci-data/Eagle_dataset/Tri")
VTKHDF_PATH = Path("/var/llm/sci-data/INS_2D_EAGLE-Drone-Wake.vtkhdf")
OUT_DIR = Path(".results/eagle/mesh_anim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 3
FRAME_SKIP = 30
N_TIMESTEPS = 990


def load_original_sample(scene_id, sim_id):
    path = DATA_ROOT / str(scene_id) / str(sim_id)
    data = np.load(path / "sim.npz")
    tri = np.load(path / "triangles.npy")
    return {
        "x": data["pointcloud"],
        "VX": data["VX"],
        "mask": data["mask"],
        "tri": tri,
    }


def load_fixed_sample(sample_idx):
    with h5py.File(VTKHDF_PATH, "r") as f:
        offset = f["SampleNodeOffset"][sample_idx]
        count = f["SampleNodeCount"][sample_idx]
        points = f["VTKHDF/Points"][offset : offset + count, :2]

        node_counts = f["SampleNodeCount"][:]
        base = sum(int(node_counts[s]) * N_TIMESTEPS for s in range(sample_idx))

        vx_all = f["VTKHDF/PointData/VX"][:]
        mask_all = f["VTKHDF/PointData/drone_mask"][:]

        vx = np.zeros((N_TIMESTEPS, count), dtype=np.float32)
        masks = np.zeros((N_TIMESTEPS, count), dtype=np.uint8)

        for t in range(N_TIMESTEPS):
            off = base + t * count
            vx[t] = vx_all[off : off + count]
            masks[t] = mask_all[off : off + count]

    return {"x": points, "VX": vx, "mask": masks}


def get_sample_ids():
    with h5py.File(VTKHDF_PATH, "r") as f:
        return f["SampleIDs"][:].tolist()


def compute_color_limits(orig, fixed):
    all_vx = np.concatenate([orig["VX"].flatten(), fixed["VX"].flatten()])
    all_vx = all_vx[np.isfinite(all_vx)]
    vmin, vmax = np.percentile(all_vx, [2, 98])
    vlim = max(abs(vmin), abs(vmax))
    return -vlim, vlim


def create_animation(orig, fixed, sample_idx, scene_id, sim_id):
    frames = list(range(0, N_TIMESTEPS, FRAME_SKIP))
    vmin, vmax = compute_color_limits(orig, fixed)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Sample {sample_idx} (scene={scene_id}, sim={sim_id})",
        fontsize=14,
        fontweight="bold",
    )

    ax_orig_mesh = axes[0, 0]
    ax_fixed_mesh = axes[0, 1]
    ax_diff_mesh = axes[0, 2]
    ax_orig_vx = axes[1, 0]
    ax_fixed_vx = axes[1, 1]
    ax_diff_vx = axes[1, 2]

    ax_orig_mesh.set_title("Original Mesh (triangles)")
    ax_fixed_mesh.set_title("Fixed Union Grid")
    ax_diff_mesh.set_title("Point Coverage")
    ax_orig_vx.set_title("Original VX")
    ax_fixed_vx.set_title("Interpolated VX")
    ax_diff_vx.set_title("Difference (Orig - Interp)")

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    all_orig_x = np.concatenate([orig["x"][t] for t in frames])
    x_min, x_max = all_orig_x[:, 0].min(), all_orig_x[:, 0].max()
    y_min, y_max = all_orig_x[:, 1].min(), all_orig_x[:, 1].max()
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    for ax in axes.flat:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    sc_fixed_mesh = ax_fixed_mesh.scatter(
        fixed["x"][:, 0], fixed["x"][:, 1], s=1, c="black", alpha=0.5
    )

    tripcolor_orig = [None]
    triplot_orig = [None]
    sc_orig_pts = [None]
    sc_fixed_pts = [None]
    tripcolor_fixed = [None]
    tripcolor_diff = [None]

    cbar_orig = [None]
    cbar_fixed = [None]
    cbar_diff = [None]

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12)

    def update(frame_idx):
        t = frames[frame_idx]

        for ax in [ax_orig_mesh, ax_orig_vx, ax_fixed_vx, ax_diff_vx, ax_diff_mesh]:
            while ax.collections:
                ax.collections[0].remove()
            while ax.lines:
                ax.lines[0].remove()

        orig_pts = orig["x"][t]
        orig_mask = orig["mask"][t].astype(bool)
        orig_tri = orig["tri"][t]
        orig_vx = orig["VX"][t]

        fluid_node_mask = ~orig_mask
        fluid_indices = np.where(fluid_node_mask)[0]
        index_map = -np.ones(len(orig_mask), dtype=np.int64)
        index_map[fluid_indices] = np.arange(len(fluid_indices))

        tri_fluid_mask = (
            fluid_node_mask[orig_tri[:, 0]]
            & fluid_node_mask[orig_tri[:, 1]]
            & fluid_node_mask[orig_tri[:, 2]]
        )
        fluid_tri = orig_tri[tri_fluid_mask]
        fluid_tri_remapped = index_map[fluid_tri]

        fluid_pts = orig_pts[fluid_node_mask]
        fluid_vx = orig_vx[fluid_node_mask]

        if len(fluid_tri_remapped) > 0 and len(fluid_pts) > 0:
            triang = Triangulation(fluid_pts[:, 0], fluid_pts[:, 1], fluid_tri_remapped)

            ax_orig_mesh.triplot(triang, "k-", lw=0.3, alpha=0.7)
            ax_orig_mesh.scatter(
                fluid_pts[:, 0], fluid_pts[:, 1], s=0.5, c="blue", alpha=0.5
            )

            ax_orig_vx.tripcolor(
                triang, fluid_vx, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="flat"
            )

        ax_orig_mesh.set_title(
            f"Original Mesh - {len(fluid_pts)} pts, {len(fluid_tri_remapped)} tri"
        )

        fixed_mask = fixed["mask"][t].astype(bool)
        fixed_fluid = ~fixed_mask
        fixed_pts = fixed["x"][fixed_fluid]
        fixed_vx = fixed["VX"][t][fixed_fluid]

        ax_fixed_vx.scatter(
            fixed_pts[:, 0],
            fixed_pts[:, 1],
            c=fixed_vx,
            s=2,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )

        ax_diff_mesh.scatter(
            fluid_pts[:, 0], fluid_pts[:, 1], s=1, c="blue", alpha=0.5, label="Original"
        )
        ax_diff_mesh.scatter(
            fixed_pts[:, 0], fixed_pts[:, 1], s=1, c="red", alpha=0.5, label="Fixed"
        )
        if frame_idx == 0:
            ax_diff_mesh.legend(loc="upper right", fontsize=8)

        from scipy.spatial import cKDTree

        if len(fluid_pts) > 0 and len(fixed_pts) > 0:
            tree = cKDTree(fluid_pts)
            dists, idx = tree.query(fixed_pts, k=1)
            close_mask = dists < 0.02

            if close_mask.sum() > 0:
                orig_at_fixed = fluid_vx[idx[close_mask]]
                interp_at_fixed = fixed_vx[close_mask]
                diff = orig_at_fixed - interp_at_fixed

                diff_lim = max(
                    abs(np.percentile(diff, 5)), abs(np.percentile(diff, 95))
                )
                if diff_lim < 1e-6:
                    diff_lim = 1.0

                ax_diff_vx.scatter(
                    fixed_pts[close_mask, 0],
                    fixed_pts[close_mask, 1],
                    c=diff,
                    s=2,
                    cmap="coolwarm",
                    vmin=-diff_lim,
                    vmax=diff_lim,
                )

                mae = np.abs(diff).mean()
                rmse = np.sqrt((diff**2).mean())
                ax_diff_vx.set_title(f"Difference | MAE={mae:.3f} RMSE={rmse:.3f}")

        time_text.set_text(f"Timestep: {t}/{N_TIMESTEPS - 1}")

        return []

    L(f"Creating animation for sample {sample_idx}...")
    anim = FuncAnimation(fig, update, frames=len(frames), interval=150, blit=False)

    out_path = OUT_DIR / f"sample_{sample_idx}_s{scene_id}_sim{sim_id}.mp4"
    writer = FFMpegWriter(fps=8, bitrate=3000)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)

    return out_path


def main():
    L("=" * 60)
    L("EAGLE Mesh Evolution Diagnostic v2")
    L("=" * 60)

    sample_ids = get_sample_ids()
    n_total = len(sample_ids)

    rng = np.random.default_rng(int(time.time()))
    chosen_indices = rng.choice(n_total, size=N_SAMPLES, replace=False)

    L(f"Samples: {n_total}, chosen: {chosen_indices}")

    for sample_idx in chosen_indices:
        scene_id, sim_id = sample_ids[sample_idx]
        L(f"Processing sample {sample_idx} (scene={scene_id}, sim={sim_id})")

        orig = load_original_sample(scene_id, sim_id)
        L(f"  Original: {orig['x'].shape}, tri: {orig['tri'][0].shape}")

        fixed = load_fixed_sample(sample_idx)
        L(f"  Fixed: {fixed['x'].shape}, VX: {fixed['VX'].shape}")

        out_path = create_animation(orig, fixed, sample_idx, scene_id, sim_id)
        L(f"  Saved: {out_path}")

    L("=" * 60)
    L(f"Done. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
