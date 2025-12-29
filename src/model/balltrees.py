import math
import torch
import triton
import triton.language as tl
from typing import List, Tuple


@triton.jit
def _compute_split_kernel(
    data_ptr,
    indices_ptr,
    proj_ptr,
    split_dims_ptr,
    pivots_ptr,
    seg_starts_ptr,
    seg_ends_ptr,
    seg_batch_ptr,
    batch_data_starts_ptr,
    n_segs,
    dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seg_id = tl.program_id(0)
    if seg_id >= n_segs:
        return

    start = tl.load(seg_starts_ptr + seg_id)
    end = tl.load(seg_ends_ptr + seg_id)
    n = end - start

    if n <= 1:
        tl.store(split_dims_ptr + seg_id, 0)
        tl.store(pivots_ptr + seg_id, 0.0)
        return

    batch_id = tl.load(seg_batch_ptr + seg_id)
    data_offset = tl.load(batch_data_starts_ptr + batch_id)

    first_idx = tl.load(indices_ptr + start)
    best_dim = 0
    best_spread = tl.load(data_ptr + (data_offset + first_idx) * dim).to(tl.float64)
    best_spread = best_spread - best_spread

    for d in range(dim):
        min_val: tl.float64 = 1e308
        max_val: tl.float64 = -1e308
        for block_start in range(0, n, BLOCK):
            offs = block_start + tl.arange(0, BLOCK)
            mask = offs < n
            local_idx = tl.load(indices_ptr + start + offs, mask=mask, other=0)
            vals = tl.load(
                data_ptr + (data_offset + local_idx) * dim + d, mask=mask, other=0.0
            ).to(tl.float64)
            min_val = tl.minimum(min_val, tl.min(tl.where(mask, vals, 1e308)))
            max_val = tl.maximum(max_val, tl.max(tl.where(mask, vals, -1e308)))
        spread = max_val - min_val
        if spread > best_spread:
            best_spread = spread
            best_dim = d

    tl.store(split_dims_ptr + seg_id, best_dim)

    sum_val = best_spread - best_spread
    for block_start in range(0, n, BLOCK):
        offs = block_start + tl.arange(0, BLOCK)
        mask = offs < n
        local_idx = tl.load(indices_ptr + start + offs, mask=mask, other=0)
        vals = tl.load(
            data_ptr + (data_offset + local_idx) * dim + best_dim, mask=mask, other=0.0
        ).to(tl.float64)
        sum_val += tl.sum(tl.where(mask, vals, 0.0))
        tl.store(proj_ptr + start + offs, vals, mask=mask)

    pivot = sum_val / n.to(tl.float64)
    tl.store(pivots_ptr + seg_id, pivot)


@triton.jit
def _write_leaves_kernel(
    indices_ptr,
    batch_data_starts_ptr,
    tree_offsets_ptr,
    seg_starts_ptr,
    seg_ends_ptr,
    seg_batch_ptr,
    leaf_ids_ptr,
    out_idx_ptr,
    out_mask_ptr,
    n_segs,
):
    seg_id = tl.program_id(0)
    if seg_id >= n_segs:
        return

    start = tl.load(seg_starts_ptr + seg_id)
    end = tl.load(seg_ends_ptr + seg_id)
    n = end - start
    batch = tl.load(seg_batch_ptr + seg_id)
    leaf_id = tl.load(leaf_ids_ptr + seg_id)

    data_offset = tl.load(batch_data_starts_ptr + batch)
    tree_offset = tl.load(tree_offsets_ptr + batch)
    out_pos = tree_offset + leaf_id * 2

    if n == 0:
        tl.store(out_idx_ptr + out_pos, 0)
        tl.store(out_idx_ptr + out_pos + 1, 0)
        tl.store(out_mask_ptr + out_pos, False)
        tl.store(out_mask_ptr + out_pos + 1, False)
    elif n == 1:
        local_idx = tl.load(indices_ptr + start)
        gidx = data_offset + local_idx
        tl.store(out_idx_ptr + out_pos, gidx)
        tl.store(out_idx_ptr + out_pos + 1, gidx)
        tl.store(out_mask_ptr + out_pos, True)
        tl.store(out_mask_ptr + out_pos + 1, False)
    else:
        li0 = tl.load(indices_ptr + start)
        li1 = tl.load(indices_ptr + start + 1)
        tl.store(out_idx_ptr + out_pos, data_offset + li0)
        tl.store(out_idx_ptr + out_pos + 1, data_offset + li1)
        tl.store(out_mask_ptr + out_pos, True)
        tl.store(out_mask_ptr + out_pos + 1, True)


def _get_batch_info(batch_idx: torch.Tensor, device: torch.device):
    n = batch_idx.shape[0]
    if n == 0:
        empty = torch.tensor([], device=device, dtype=torch.int64)
        return empty, empty, empty, empty, empty, empty, 0

    batch_ids, inverse = torch.unique(batch_idx, sorted=True, return_inverse=True)
    n_batches = batch_ids.shape[0]
    counts = torch.bincount(inverse, minlength=n_batches)

    ends = counts.cumsum(0)
    starts = torch.cat([torch.zeros(1, device=device, dtype=torch.int64), ends[:-1]])

    max_levels = (counts.float().log2().ceil() - 1).clamp(min=0).long()
    n_leaves = (1 << max_levels).long()
    tree_sizes = n_leaves * 2

    tree_offsets = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int64), tree_sizes.cumsum(0)[:-1]]
    )
    total_tree_size = tree_sizes.sum().item()

    return starts, ends, counts, max_levels, tree_offsets, tree_sizes, total_tree_size


def _partition_and_split(
    proj: torch.Tensor,
    indices: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_ends: torch.Tensor,
    seg_batch: torch.Tensor,
    pivots: torch.Tensor,
    active: torch.Tensor,
):
    device = proj.device
    n = proj.shape[0]
    n_segs = seg_starts.shape[0]

    if n == 0 or n_segs == 0:
        empty = torch.tensor([], device=device, dtype=torch.int64)
        return empty, empty, empty

    active_bool = active.bool()
    seg_lengths = seg_ends - seg_starts

    point_seg = (
        torch.bucketize(torch.arange(n, device=device), seg_starts, right=True) - 1
    )
    point_seg = point_seg.clamp(0, n_segs - 1)

    max_proj = proj.abs().max() + 1
    sort_key = point_seg.double() * max_proj * 2 + proj.double()
    sorted_order = sort_key.argsort()

    new_indices = indices[sorted_order]
    indices.copy_(new_indices)

    left_counts = seg_lengths // 2
    left_counts = torch.where(active_bool & (seg_lengths > 1), left_counts, seg_lengths)
    left_counts = torch.clamp(left_counts, min=1)
    left_counts = torch.where(seg_lengths <= 1, seg_lengths, left_counts)

    mids = seg_starts + left_counts

    n_out = n_segs * 2
    new_starts = torch.empty(n_out, device=device, dtype=torch.int64)
    new_ends = torch.empty(n_out, device=device, dtype=torch.int64)
    new_batch = torch.empty(n_out, device=device, dtype=torch.int64)

    should_split = active_bool & (seg_lengths > 1)

    new_starts[0::2] = seg_starts
    new_ends[0::2] = torch.where(should_split, mids, seg_ends)
    new_starts[1::2] = torch.where(should_split, mids, seg_ends)
    new_ends[1::2] = seg_ends
    new_batch[0::2] = seg_batch
    new_batch[1::2] = seg_batch

    valid = (new_ends - new_starts) > 0
    return new_starts[valid], new_ends[valid], new_batch[valid]


def build_balltree(
    data: torch.Tensor, batch_idx: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = data.device
    n, dim = data.shape
    dtype = data.dtype

    if n == 0:
        return (
            torch.tensor([], device=device, dtype=torch.int64),
            torch.tensor([], device=device, dtype=torch.bool),
        )

    starts, ends, counts, max_levels, tree_offsets, tree_sizes, total_tree_size = (
        _get_batch_info(batch_idx, device)
    )
    n_batches = starts.shape[0]

    if n_batches == 0:
        return (
            torch.tensor([], device=device, dtype=torch.int64),
            torch.tensor([], device=device, dtype=torch.bool),
        )

    global_max_level = max_levels.max().item()

    indices = torch.empty(n, device=device, dtype=torch.int64)
    for i in range(n_batches):
        s, e = starts[i].item(), ends[i].item()
        indices[s:e] = torch.arange(e - s, device=device, dtype=torch.int64)

    proj = torch.empty(n, device=device, dtype=dtype)

    seg_starts = starts.clone()
    seg_ends = ends.clone()
    seg_batch = torch.arange(n_batches, device=device, dtype=torch.int64)
    current_level = torch.zeros(n_batches, device=device, dtype=torch.int64)

    BLOCK = 256

    for level in range(global_max_level):
        n_segs = seg_starts.shape[0]
        if n_segs == 0:
            break

        seg_levels = current_level[seg_batch]
        active = (seg_levels < max_levels[seg_batch]).to(torch.int32)

        split_dims = torch.empty(n_segs, device=device, dtype=torch.int64)
        pivots = torch.empty(n_segs, device=device, dtype=dtype)

        _compute_split_kernel[(n_segs,)](
            data,
            indices,
            proj,
            split_dims,
            pivots,
            seg_starts,
            seg_ends,
            seg_batch,
            starts,
            n_segs,
            dim,
            BLOCK=BLOCK,
        )

        seg_starts, seg_ends, seg_batch = _partition_and_split(
            proj, indices, seg_starts, seg_ends, seg_batch, pivots, active
        )

        current_level += 1

    n_segs = seg_starts.shape[0]

    if n_segs == 0:
        return (
            torch.zeros(total_tree_size, device=device, dtype=torch.int64),
            torch.zeros(total_tree_size, device=device, dtype=torch.bool),
        )

    seg_batch_sorted, batch_order = seg_batch.sort()
    leaf_ids = torch.zeros(n_segs, device=device, dtype=torch.int64)
    batch_counts = torch.bincount(seg_batch, minlength=n_batches)
    batch_offsets = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int64), batch_counts.cumsum(0)[:-1]]
    )
    positions_in_batch = (
        torch.arange(n_segs, device=device) - batch_offsets[seg_batch_sorted]
    )
    leaf_ids[batch_order] = positions_in_batch

    out_idx = torch.zeros(total_tree_size, device=device, dtype=torch.int64)
    out_mask = torch.zeros(total_tree_size, device=device, dtype=torch.bool)

    _write_leaves_kernel[(n_segs,)](
        indices,
        starts,
        tree_offsets,
        seg_starts,
        seg_ends,
        seg_batch,
        leaf_ids,
        out_idx,
        out_mask,
        n_segs,
    )

    return out_idx, out_mask


def partition_balltree(
    data: torch.Tensor, batch_idx: torch.Tensor, target_level: int
) -> torch.Tensor:
    device = data.device
    n, dim = data.shape
    dtype = data.dtype

    if n == 0 or target_level <= 0:
        return torch.arange(n, device=device, dtype=torch.int64)

    starts, ends, counts, _, _, _, _ = _get_batch_info(batch_idx, device)
    n_batches = starts.shape[0]

    if n_batches == 0:
        return torch.arange(n, device=device, dtype=torch.int64)

    indices = torch.arange(n, device=device, dtype=torch.int64)
    proj = torch.empty(n, device=device, dtype=dtype)

    seg_starts = starts.clone()
    seg_ends = ends.clone()
    seg_batch = torch.arange(n_batches, device=device, dtype=torch.int64)

    BLOCK = 256

    for level in range(target_level):
        n_segs = seg_starts.shape[0]
        if n_segs == 0:
            break

        split_dims = torch.empty(n_segs, device=device, dtype=torch.int64)
        pivots = torch.empty(n_segs, device=device, dtype=dtype)

        _compute_split_kernel[(n_segs,)](
            data,
            indices,
            proj,
            split_dims,
            pivots,
            seg_starts,
            seg_ends,
            seg_batch,
            starts,
            n_segs,
            dim,
            BLOCK=BLOCK,
        )

        active = torch.ones(n_segs, device=device, dtype=torch.int32)
        seg_starts, seg_ends, seg_batch = _partition_and_split(
            proj, indices, seg_starts, seg_ends, seg_batch, pivots, active
        )

    return indices


def generate_rotation_matrix(
    angle: float, dim: int, device: torch.device
) -> torch.Tensor:
    angle_rad = math.radians(angle)
    c, s = math.cos(angle_rad), math.sin(angle_rad)

    if dim == 2:
        return torch.tensor([[c, -s], [s, c]], device=device, dtype=torch.float32)
    elif dim == 3:
        return torch.tensor(
            [
                [c * c, s * c * (s - 1), s * (s + c * c)],
                [s * c, s * s * s + c * c, s * c * (s - 1)],
                [-s, s * c, c * c],
            ],
            device=device,
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


@torch.compiler.disable(recursive=False)
def build_balltree_with_rotations(
    data: torch.Tensor,
    batch_idx: torch.Tensor,
    strides: List[int],
    ball_sizes: List[int],
    angle: float = 45.0,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    dim = data.shape[1]
    num_layers = len(ball_sizes)

    tree_idx, tree_mask = build_balltree(data, batch_idx)

    if angle <= 0:
        return tree_idx, tree_mask, [None] * num_layers

    leaves = data[tree_idx]
    current_batch_idx = batch_idx[tree_idx]

    rotation_matrix = generate_rotation_matrix(angle, dim, data.device)
    rotated_leaves = torch.matmul(leaves, rotation_matrix)

    rot_tree_indices = []
    for i in range(num_layers):
        n_points = rotated_leaves.shape[0]
        target_partitions = (
            max(0, int(math.log2(n_points / ball_sizes[i]))) if ball_sizes[i] > 0 else 0
        )

        rot_idx = partition_balltree(
            rotated_leaves, current_batch_idx, target_partitions
        )
        rot_tree_indices.append(rot_idx)

        if i < num_layers - 1:
            stride = strides[i]
            n_coarse = rotated_leaves.shape[0] // stride
            if n_coarse > 0:
                rotated_leaves = (
                    rotated_leaves[: n_coarse * stride]
                    .view(n_coarse, stride, dim)
                    .mean(dim=1)
                )
                current_batch_idx = current_batch_idx[::stride]

    return tree_idx, tree_mask, rot_tree_indices
