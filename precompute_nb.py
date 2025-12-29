"""
Precompute neighbors for EAGLE dataset.
Coordinates normalized to [-1, 1], latent grid in [-1, 1].
"""
import numpy as np
import h5py
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from src.model.layers.utils.neighbor_search import NeighborSearch
import torch

VTKHDF_PATH = "/var/llm/sci-data/INS_2D_EAGLE-Drone-Wake.vtkhdf"
OUTPUT_PATH = "/var/llm/sci-data/eagle_neighbors.pt"
LATENT_SIZE = [64, 64]
BASE_RADIUS = 0.05
SCALES = [1.0, 2.0, 4.0]

with h5py.File(VTKHDF_PATH, "r") as f:
    POINTS = f["VTKHDF/Points"][:, :2].astype(np.float32)
    OFFSETS = f["SampleNodeOffset"][:].astype(np.int64)
    COUNTS = f["SampleNodeCount"][:].astype(np.int64)

COORD_MIN = POINTS.min(axis=0)
COORD_MAX = POINTS.max(axis=0)

LATENT_GRID = np.stack(np.meshgrid(
    np.linspace(-1, 1, LATENT_SIZE[0]),
    np.linspace(-1, 1, LATENT_SIZE[1]),
    indexing='ij'
), axis=-1).reshape(-1, 2).astype(np.float32)


def normalize_coords(coords):
    return 2.0 * (coords - COORD_MIN) / (COORD_MAX - COORD_MIN) - 1.0


def compute_neighbors(i):
    ns = NeighborSearch(method='native')
    start, count = OFFSETS[i], COUNTS[i]
    coords = POINTS[start:start + count]
    coords_norm = normalize_coords(coords)
    
    coords_t = torch.from_numpy(coords_norm)
    latent_t = torch.from_numpy(LATENT_GRID)
    
    enc_scales, dec_scales = [], []
    for scale in SCALES:
        r = BASE_RADIUS * scale
        enc = ns(data=coords_t, queries=latent_t, radius=r)
        dec = ns(data=latent_t, queries=coords_t, radius=r)
        enc_scales.append({k: v.numpy() for k, v in enc.items()})
        dec_scales.append({k: v.numpy() for k, v in dec.items()})
    
    return i, enc_scales, dec_scales


if __name__ == "__main__":
    n_samples = len(COUNTS)
    n_workers = min(cpu_count(), 128)
    
    print(f"EAGLE Neighbor Precomputation")
    print(f"=" * 50)
    print(f"Input:        {VTKHDF_PATH}")
    print(f"Output:       {OUTPUT_PATH}")
    print(f"Samples:      {n_samples}")
    print(f"Workers:      {n_workers}")
    print(f"Latent grid:  {LATENT_SIZE[0]}x{LATENT_SIZE[1]} = {LATENT_GRID.shape[0]} points")
    print(f"Base radius:  {BASE_RADIUS}")
    print(f"Scales:       {SCALES}")
    print(f"Coord range:  [{COORD_MIN}] to [{COORD_MAX}]")
    print(f"=" * 50)

    encoder_nbrs = [None] * n_samples
    decoder_nbrs = [None] * n_samples

    with Pool(n_workers) as pool:
        for i, enc, dec in tqdm(pool.imap_unordered(compute_neighbors, range(n_samples)), total=n_samples):
            encoder_nbrs[i] = enc
            decoder_nbrs[i] = dec

    print("\nSaving...")
    torch.save({
        'encoder_nbrs': encoder_nbrs,
        'decoder_nbrs': decoder_nbrs,
        'latent_grid': LATENT_GRID,
        'coord_min': COORD_MIN,
        'coord_max': COORD_MAX,
        'scales': SCALES,
        'base_radius': BASE_RADIUS,
        'latent_size': LATENT_SIZE,
    }, OUTPUT_PATH)
    
    print(f"Saved: {OUTPUT_PATH}")
    print("Done!")
