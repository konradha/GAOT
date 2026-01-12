"""
EAGLE Dataset Converter: Per-Sample Union Grid → VTKHDF
"""

import numpy as np
import torch
import h5py
from pathlib import Path
from scipy.spatial import Delaunay, cKDTree
from tqdm import tqdm
import time

DATA_ROOT = Path("/var/llm/sci-data/Eagle_dataset/Tri")
OUT_PATH = Path("/var/llm/sci-data/INS_2D_EAGLE-Drone-Wake.vtkhdf")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sample(scene_id, sim_id):
    path = DATA_ROOT / str(scene_id) / str(sim_id)
    data = np.load(path / "sim.npz")
    tri = np.load(path / "triangles.npy")
    return {
        "x": data["pointcloud"],
        "VX": data["VX"],
        "VY": data["VY"],
        "PS": data["PS"],
        "PG": data["PG"],
        "mask": data["mask"],
        "tri": tri,
    }


def get_available_samples():
    samples = []
    for scene_dir in DATA_ROOT.iterdir():
        if not scene_dir.is_dir():
            continue
        for sim_dir in scene_dir.iterdir():
            if not sim_dir.is_dir():
                continue
            if (sim_dir / "sim.npz").exists():
                samples.append((int(scene_dir.name), int(sim_dir.name)))
    return sorted(samples)


def validate_sample(sample):
    for t in range(sample["x"].shape[0]):
        if sample["tri"][t].max() >= sample["x"][t].shape[0]:
            return False
    return True


def build_sample_union_grid(sample, merge_tol):
    n_timesteps = sample["x"].shape[0]
    inv_tol = 1.0 / merge_tol

    point_set = {}
    for t in range(n_timesteps):
        fluid_mask = ~sample["mask"][t].astype(bool)
        fluid_pts = sample["x"][t][fluid_mask]
        keys = (fluid_pts * inv_tol).astype(np.int32)
        for i in range(len(keys)):
            key = (keys[i, 0], keys[i, 1])
            if key not in point_set:
                point_set[key] = fluid_pts[i]

    return np.array(list(point_set.values()), dtype=np.float32)


@torch.no_grad()
def interpolate_timestep_gpu_fast(
    query_pts, source_x, source_tri, source_mask, source_fields, device
):
    n_query = len(query_pts)
    var_names = list(source_fields.keys())

    query_t = torch.from_numpy(query_pts).float().to(device)
    source_x_t = torch.from_numpy(source_x).float().to(device)
    source_tri_t = torch.from_numpy(source_tri.astype(np.int64)).to(device)
    source_mask_t = torch.from_numpy(source_mask.astype(bool)).to(device)

    v0 = source_x_t[source_tri_t[:, 0]]
    v1 = source_x_t[source_tri_t[:, 1]]
    v2 = source_x_t[source_tri_t[:, 2]]

    m0 = source_mask_t[source_tri_t[:, 0]]
    m1 = source_mask_t[source_tri_t[:, 1]]
    m2 = source_mask_t[source_tri_t[:, 2]]
    tri_fluid = ~(m0 | m1 | m2)
    tri_drone = m0 & m1 & m2

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    d00 = (v0v1 * v0v1).sum(dim=1)
    d01 = (v0v1 * v0v2).sum(dim=1)
    d11 = (v0v2 * v0v2).sum(dim=1)
    denom = d00 * d11 - d01 * d01
    denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
    inv_denom = 1.0 / denom

    field_tensors = {
        vn: torch.from_numpy(f.astype(np.float32)).to(device)
        for vn, f in source_fields.items()
    }
    f0 = {vn: field_tensors[vn][source_tri_t[:, 0]] for vn in var_names}
    f1 = {vn: field_tensors[vn][source_tri_t[:, 1]] for vn in var_names}
    f2 = {vn: field_tensors[vn][source_tri_t[:, 2]] for vn in var_names}

    result_fields = {
        vn: torch.zeros(n_query, device=device, dtype=torch.float32) for vn in var_names
    }
    result_mask = torch.zeros(n_query, device=device, dtype=torch.uint8)
    in_fluid = torch.zeros(n_query, device=device, dtype=torch.bool)

    chunk_size = 4096
    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        chunk_pts = query_t[start:end]

        v0p = chunk_pts[:, None, :] - v0[None, :, :]
        d02 = (v0v1[None, :, :] * v0p).sum(dim=2)
        d12 = (v0v2[None, :, :] * v0p).sum(dim=2)

        u = (d11 * d02 - d01 * d12) * inv_denom
        v = (d00 * d12 - d01 * d02) * inv_denom
        w = 1.0 - u - v

        inside = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6)
        inside_fluid = inside & tri_fluid[None, :]
        inside_drone = inside & tri_drone[None, :]

        any_fluid = inside_fluid.any(dim=1)
        any_drone = inside_drone.any(dim=1) & ~any_fluid

        result_mask[start:end][any_drone] = 1
        in_fluid[start:end] = any_fluid

        if any_fluid.sum() == 0:
            continue

        first_fluid_tri = inside_fluid.float().argmax(dim=1)

        fluid_idx = torch.where(any_fluid)[0]
        tri_idx = first_fluid_tri[any_fluid]

        w_vals = w[any_fluid, tri_idx]
        u_vals = u[any_fluid, tri_idx]
        v_vals = v[any_fluid, tri_idx]

        for vn in var_names:
            interp = (
                w_vals * f0[vn][tri_idx]
                + u_vals * f1[vn][tri_idx]
                + v_vals * f2[vn][tri_idx]
            )
            result_fields[vn][start:end][any_fluid] = interp

    return (
        {vn: result_fields[vn].cpu().numpy() for vn in var_names},
        result_mask.cpu().numpy(),
        in_fluid.cpu().numpy(),
    )


def process_sample_gpu(sample, merge_tol, device):
    n_timesteps = sample["x"].shape[0]
    var_names = ["VX", "VY", "PS", "PG"]

    union_grid = build_sample_union_grid(sample, merge_tol)
    n_nodes = len(union_grid)

    fields = {
        vn: np.zeros((n_timesteps, n_nodes), dtype=np.float32) for vn in var_names
    }
    masks = np.zeros((n_timesteps, n_nodes), dtype=np.uint8)

    for t in range(n_timesteps):
        source_fields = {vn: sample[vn][t] for vn in var_names}

        t_fields, t_mask, _ = interpolate_timestep_gpu_fast(
            union_grid,
            sample["x"][t],
            sample["tri"][t],
            sample["mask"][t],
            source_fields,
            device,
        )

        for vn in var_names:
            fields[vn][t] = t_fields[vn]
        masks[t] = t_mask

    triangles = Delaunay(union_grid).simplices.astype(np.int64)

    return {
        "grid": union_grid,
        "triangles": triangles,
        "fields": fields,
        "masks": masks,
        "n_timesteps": n_timesteps,
    }


def compute_merge_tolerance(all_samples):
    print("Computing merge tolerance...")
    all_edges = []

    for scene_id, sim_id in tqdm(all_samples[:30], desc="Sampling edges"):
        try:
            sample = load_sample(scene_id, sim_id)
            if not validate_sample(sample):
                continue
            pts = sample["x"][0]
            tri = sample["tri"][0]
            v0, v1, v2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
            edges = np.concatenate(
                [
                    np.linalg.norm(v1 - v0, axis=1),
                    np.linalg.norm(v2 - v1, axis=1),
                    np.linalg.norm(v0 - v2, axis=1),
                ]
            )
            all_edges.append(edges)
        except:
            continue

    all_edges = np.concatenate(all_edges)
    median_edge = np.median(all_edges)
    merge_tol = median_edge * 0.5

    print(f"  Median edge: {median_edge:.6f}")
    print(f"  Merge tolerance: {merge_tol:.6f}")
    return merge_tol


def compute_errors(union_grid, sample, fields):
    var_names = ["VX", "VY", "PS", "PG"]
    tree = cKDTree(union_grid)
    errors = {vn: [] for vn in var_names}

    for t in [0, sample["x"].shape[0] // 2, sample["x"].shape[0] - 1]:
        fluid_mask = ~sample["mask"][t].astype(bool)
        fluid_pts = sample["x"][t][fluid_mask]
        dists, idx = tree.query(fluid_pts, k=1)
        close = dists < 0.005

        if close.sum() == 0:
            continue

        for vn in var_names:
            orig = sample[vn][t][fluid_mask][close]
            interp = fields[vn][t][idx[close]]
            valid = interp != 0
            if valid.sum() > 0:
                diff = np.abs(orig[valid] - interp[valid])
                errors[vn].append(
                    {
                        "l1": diff.mean(),
                        "l2": np.sqrt((diff**2).mean()),
                        "linf": diff.max(),
                    }
                )

    return {
        vn: {
            "l1": np.mean([e["l1"] for e in errs]),
            "l2": np.mean([e["l2"] for e in errs]),
            "linf": np.max([e["linf"] for e in errs]),
        }
        for vn, errs in errors.items()
        if errs
    }


def write_vtkhdf(path, all_results, sample_ids):
    n_samples = len(all_results)
    n_timesteps = all_results[0]["n_timesteps"]
    total_nodes = sum(len(r["grid"]) for r in all_results)
    total_triangles = sum(len(r["triangles"]) for r in all_results)

    print(
        f"\nWriting VTKHDF: {n_samples} samples, {total_nodes:,} nodes, {total_triangles:,} triangles"
    )

    sample_node_offsets = np.cumsum([0] + [len(r["grid"]) for r in all_results]).astype(
        np.int64
    )
    sample_tri_offsets = np.cumsum(
        [0] + [len(r["triangles"]) for r in all_results]
    ).astype(np.int64)

    with h5py.File(path, "w") as f:
        vhdf = f.create_group("VTKHDF")
        vhdf.attrs["Type"] = np.bytes_("UnstructuredGrid")
        vhdf.attrs["Version"] = np.array([2, 0], dtype=np.int64)

        print("  Points...")
        pts = np.zeros((total_nodes, 3), dtype=np.float32)
        for i, r in enumerate(all_results):
            pts[sample_node_offsets[i] : sample_node_offsets[i + 1], :2] = r["grid"]
        vhdf.create_dataset("Points", data=pts, compression="gzip", compression_opts=1)
        del pts

        print("  Connectivity...")
        conn = np.concatenate(
            [
                (r["triangles"] + sample_node_offsets[i]).flatten()
                for i, r in enumerate(all_results)
            ]
        ).astype(np.int64)
        vhdf.create_dataset(
            "Connectivity", data=conn, compression="gzip", compression_opts=1
        )
        del conn

        vhdf.create_dataset(
            "Offsets",
            data=(np.arange(1, total_triangles + 1) * 3).astype(np.int64),
            compression="gzip",
            compression_opts=1,
        )
        vhdf.create_dataset(
            "Types",
            data=np.full(total_triangles, 5, dtype=np.uint8),
            compression="gzip",
            compression_opts=1,
        )
        vhdf.create_dataset(
            "NumberOfPoints", data=np.array([total_nodes], dtype=np.int64)
        )
        vhdf.create_dataset(
            "NumberOfCells", data=np.array([total_triangles], dtype=np.int64)
        )
        vhdf.create_dataset(
            "NumberOfConnectivityIds",
            data=np.array([total_triangles * 3], dtype=np.int64),
        )

        pdata = vhdf.create_group("PointData")
        for vn in ["VX", "VY", "PS", "PG"]:
            print(f"  {vn}...")
            arr = np.concatenate([r["fields"][vn].ravel() for r in all_results])
            pdata.create_dataset(vn, data=arr, compression="gzip", compression_opts=1)
            del arr

        print("  drone_mask...")
        arr = np.concatenate([r["masks"].ravel() for r in all_results]).astype(np.uint8)
        pdata.create_dataset(
            "drone_mask", data=arr, compression="gzip", compression_opts=1
        )
        del arr

        print("  Steps...")
        steps = vhdf.create_group("Steps")
        total_timesteps = n_samples * n_timesteps

        time_vals = np.tile(np.arange(n_timesteps, dtype=np.float64), n_samples)
        point_offs = []
        for i, r in enumerate(all_results):
            base = sample_node_offsets[i]
            n = len(r["grid"])
            point_offs.extend([base + t * n for t in range(n_timesteps)])

        steps.create_dataset("Values", data=time_vals)
        steps.create_dataset(
            "NumberOfParts", data=np.ones(total_timesteps, dtype=np.int64)
        )
        steps.create_dataset(
            "PointOffsets", data=np.array(point_offs, dtype=np.int64).reshape(-1, 1)
        )
        steps.create_dataset(
            "CellOffsets", data=np.zeros((total_timesteps, 1), dtype=np.int64)
        )
        steps.create_dataset(
            "ConnectivityIdOffsets", data=np.zeros((total_timesteps, 1), dtype=np.int64)
        )

        f.create_dataset(
            "SampleOffset", data=(np.arange(n_samples) * n_timesteps).astype(np.int64)
        )
        f.create_dataset("SampleNodeOffset", data=sample_node_offsets[:-1])
        f.create_dataset("SampleNodeCount", data=np.diff(sample_node_offsets))
        f.create_dataset("SampleTriangleCount", data=np.diff(sample_tri_offsets))
        f.create_dataset(
            "SampleIDs",
            data=np.array([[s[0], s[1]] for s in sample_ids], dtype=np.int32),
        )

    print(f"  Size: {path.stat().st_size / 1e9:.2f} GB")


def main():
    t0 = time.time()

    print("=" * 60)
    print("EAGLE → VTKHDF (Per-Sample Union, GPU)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    all_samples = get_available_samples()
    print(f"Found {len(all_samples)} samples")

    merge_tol = compute_merge_tolerance(all_samples)

    all_results = []
    sample_ids = []
    error_idx = set(
        np.random.choice(len(all_samples), min(5, len(all_samples)), replace=False)
    )
    all_errors = []

    for i, (scene_id, sim_id) in enumerate(tqdm(all_samples, desc="Processing")):
        try:
            sample = load_sample(scene_id, sim_id)
            if not validate_sample(sample):
                continue

            result = process_sample_gpu(sample, merge_tol, DEVICE)
            all_results.append(result)
            sample_ids.append((scene_id, sim_id))

            if i in error_idx:
                errs = compute_errors(result["grid"], sample, result["fields"])
                all_errors.append(
                    {
                        "sample": (scene_id, sim_id),
                        "n_nodes": len(result["grid"]),
                        "errors": errs,
                    }
                )

            del sample
        except Exception as e:
            print(f"  Error {scene_id}/{sim_id}: {e}")

    print(f"\nProcessed {len(all_results)} samples")
    node_counts = [len(r["grid"]) for r in all_results]
    print(
        f"Nodes: {min(node_counts)}-{max(node_counts)}, mean={np.mean(node_counts):.0f}"
    )

    write_vtkhdf(OUT_PATH, all_results, sample_ids)

    print("\n" + "=" * 60)
    print("ERRORS")
    for e in all_errors:
        print(f"{e['sample']} ({e['n_nodes']} nodes):")
        for vn, err in e["errors"].items():
            print(
                f"  {vn}: L1={err['l1']:.4f}, L2={err['l2']:.4f}, Linf={err['linf']:.4f}"
            )

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    print(f"Output: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
