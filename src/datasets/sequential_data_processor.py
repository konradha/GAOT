"""
Sequential data processing utilities for time-dependent GAOT datasets.
Extends the base DataProcessor for temporal data handling.
With mask support for variable-validity point clouds (e.g., EAGLE).
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader

from .data_processor import DataProcessor, EPSILON
from ..core.trainer_utils import compute_data_stats, normalize_data
from ..core.trainer_utils import compute_sequential_stats


class SequentialDataProcessor(DataProcessor):
    def __init__(self, dataset_config, metadata, dtype: torch.dtype = torch.float32):
        super().__init__(dataset_config, metadata, dtype)

        self.t_values = None
        self.stats = None

        self.max_time_diff = dataset_config.max_time_diff
        self.time_step = dataset_config.time_step
        self.stepper_mode = dataset_config.stepper_mode
        self.use_time_norm = dataset_config.use_time_norm
        self.use_metadata_stats = dataset_config.use_metadata_stats
        self.sample_rate = dataset_config.sample_rate

        self.has_mask = (
            hasattr(metadata, "group_mask") and metadata.group_mask is not None
        )

    def load_and_process_data(self) -> Tuple[Dict, bool]:
        print("Loading and preprocessing sequential data...")

        raw_data = self._load_raw_sequential_data()

        is_variable_coords = self._determine_coordinate_mode(raw_data)

        data_splits = self._split_and_normalize_sequential_data(
            raw_data, is_variable_coords
        )

        print("Sequential data loading and preprocessing complete.")
        return data_splits, is_variable_coords

    def _load_raw_sequential_data(self) -> Dict:
        import xarray as xr
        import os

        base_path = self.dataset_config.base_path
        dataset_name = self.dataset_config.name
        dataset_path = os.path.join(base_path, f"{dataset_name}.nc")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        with xr.open_dataset(dataset_path) as ds:
            u_array = ds[self.metadata.group_u].values

            if self.metadata.group_c is not None:
                c_array = ds[self.metadata.group_c].values
            else:
                c_array = None

            x_array = self._load_sequential_coordinate_data(ds, u_array)

            if self.has_mask:
                mask_array = ds[self.metadata.group_mask].values
            else:
                mask_array = None

            if self.metadata.domain_t is not None:
                t_start, t_end = self.metadata.domain_t
                self.t_values = np.linspace(t_start, t_end, u_array.shape[1])
            else:
                raise ValueError(
                    "metadata.domain_t is None. Cannot compute actual time values."
                )

        if (
            self.dataset_config.name in self.poseidon_datasets
            and self.dataset_config.use_sparse
        ):
            u_array = u_array[:, :, :9216, :]
            if c_array is not None:
                c_array = c_array[:, :, :9216, :]
            x_array = x_array[:, :, :9216, :]
            if mask_array is not None:
                mask_array = mask_array[:, :, :9216]

        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]

        return {
            "u": u_array,
            "c": c_array,
            "x": x_array,
            "t": self.t_values,
            "mask": mask_array,
        }

    def _load_sequential_coordinate_data(self, ds, u_array: np.ndarray) -> np.ndarray:
        if self.metadata.group_x is not None:
            x_array = ds[self.metadata.group_x].values

            if self.metadata.fix_x:
                if x_array.ndim == 2:
                    x_array = x_array[None, None, ...]
                elif x_array.ndim == 3:
                    x_array = x_array[:, None, ...]
            else:
                if x_array.shape[0] != u_array.shape[0]:
                    raise ValueError(
                        "Variable coordinates must have same number of samples as u_array"
                    )

        else:
            domain_x = self.metadata.domain_x
            if u_array.ndim == 4:
                num_nodes = u_array.shape[2]
                grid_size = int(np.sqrt(num_nodes))
                if grid_size * grid_size != num_nodes:
                    raise ValueError(
                        f"Cannot create square grid from {num_nodes} nodes"
                    )

                x_min, y_min = domain_x[0]
                x_max, y_max = domain_x[1]
                x_lin = np.linspace(x_min, x_max, grid_size)
                y_lin = np.linspace(y_min, y_max, grid_size)
                xv, yv = np.meshgrid(x_lin, y_lin, indexing="ij")
                x_array = np.stack([xv, yv], axis=-1).reshape(-1, 2)
                x_array = x_array[None, None, ...]
            else:
                raise ValueError(f"Unexpected u_array shape: {u_array.shape}")

        return x_array

    def _split_and_normalize_sequential_data(
        self, raw_data: Dict, is_variable_coords: bool
    ) -> Dict:
        u_array = raw_data["u"]
        c_array = raw_data["c"]
        x_array = raw_data["x"]
        t_values = raw_data["t"]
        mask_array = raw_data["mask"]

        if self.max_time_diff is not None:
            max_timesteps = self.max_time_diff + 1
            u_array = u_array[:, :max_timesteps, :, :]
            if c_array is not None:
                c_array = c_array[:, :max_timesteps, :, :]
            if is_variable_coords and x_array.shape[1] > 1:
                x_array = x_array[:, :max_timesteps, :, :]
            if mask_array is not None and mask_array.shape[1] > 1:
                mask_array = mask_array[:, :max_timesteps, :]
            t_values = t_values[:max_timesteps]
            self.t_values = t_values

        train_indices, val_indices, test_indices = self._get_split_indices(
            u_array.shape[0]
        )

        u_train = np.ascontiguousarray(u_array[train_indices])
        u_val = np.ascontiguousarray(u_array[val_indices])
        u_test = np.ascontiguousarray(u_array[test_indices])

        if c_array is not None:
            c_train = np.ascontiguousarray(c_array[train_indices])
            c_val = np.ascontiguousarray(c_array[val_indices])
            c_test = np.ascontiguousarray(c_array[test_indices])
        else:
            c_train = c_val = c_test = None

        if mask_array is not None:
            mask_train = np.ascontiguousarray(mask_array[train_indices])
            mask_val = np.ascontiguousarray(mask_array[val_indices])
            mask_test = np.ascontiguousarray(mask_array[test_indices])
        else:
            mask_train = mask_val = mask_test = None

        if is_variable_coords:
            x_train = np.ascontiguousarray(x_array[train_indices])
            x_val = np.ascontiguousarray(x_array[val_indices])
            x_test = np.ascontiguousarray(x_array[test_indices])
        else:
            x_coord = x_array[0, 0]
            x_train = x_val = x_test = x_coord

        self.stats = self._compute_sequential_stats_with_mask(
            u_train, c_train, t_values, mask_train
        )

        for key, value in self.stats.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    self.stats[key][k] = torch.tensor(v, dtype=self.dtype)

        data_splits = self._convert_to_tensors_with_mask(
            u_train,
            u_val,
            u_test,
            c_train,
            c_val,
            c_test,
            x_train,
            x_val,
            x_test,
            mask_train,
            mask_val,
            mask_test,
            is_variable_coords,
        )

        data_splits["train"]["t"] = torch.tensor(t_values, dtype=self.dtype)
        data_splits["val"]["t"] = torch.tensor(t_values, dtype=self.dtype)
        data_splits["test"]["t"] = torch.tensor(t_values, dtype=self.dtype)

        return data_splits

    def _compute_sequential_stats_with_mask(
        self,
        u_train: np.ndarray,
        c_train: Optional[np.ndarray],
        t_values: np.ndarray,
        mask_train: Optional[np.ndarray],
    ) -> Dict:
        stats = {}

        if mask_train is not None:
            mask_expanded = mask_train[..., np.newaxis]
            u_masked = np.where(mask_expanded, u_train, np.nan)
            u_flat = u_masked.reshape(-1, u_train.shape[-1])
            u_mean = np.nanmean(u_flat, axis=0)
            u_std = np.nanstd(u_flat, axis=0) + EPSILON
        else:
            u_flat = u_train.reshape(-1, u_train.shape[-1])
            u_mean = np.mean(u_flat, axis=0)
            u_std = np.std(u_flat, axis=0) + EPSILON

        stats["u"] = {"mean": u_mean, "std": u_std}

        if c_train is not None:
            if mask_train is not None:
                mask_expanded = mask_train[..., np.newaxis]
                c_masked = np.where(mask_expanded, c_train, np.nan)
                c_flat = c_masked.reshape(-1, c_train.shape[-1])
                c_mean = np.nanmean(c_flat, axis=0)
                c_std = np.nanstd(c_flat, axis=0) + EPSILON
            else:
                c_flat = c_train.reshape(-1, c_train.shape[-1])
                c_mean = np.mean(c_flat, axis=0)
                c_std = np.std(c_flat, axis=0) + EPSILON
            stats["c"] = {"mean": c_mean, "std": c_std}

        if self.use_time_norm:
            t_in_indices, t_out_indices = [], []
            for lag in range(self.time_step, self.max_time_diff + 1, self.time_step):
                for i in range(0, self.max_time_diff - lag + 1, self.time_step):
                    t_in_indices.append(i)
                    t_out_indices.append(i + lag)

            t_in_indices = np.array(t_in_indices)
            t_out_indices = np.array(t_out_indices)

            start_times = t_values[t_in_indices]
            time_diffs = t_values[t_out_indices] - t_values[t_in_indices]

            stats["start_time"] = {
                "mean": np.mean(start_times),
                "std": np.std(start_times) + EPSILON,
            }
            stats["time_diffs"] = {
                "mean": np.mean(time_diffs),
                "std": np.std(time_diffs) + EPSILON,
            }

        residuals = []
        derivatives = []

        n_samples_subset = min(int(len(u_train) * self.sample_rate), len(u_train))
        u_subset = u_train[:n_samples_subset]
        mask_subset = mask_train[:n_samples_subset] if mask_train is not None else None

        for sample_idx in range(n_samples_subset):
            for t_idx in range(min(self.max_time_diff, u_subset.shape[1] - 1)):
                u_curr = u_subset[sample_idx, t_idx]
                u_next = u_subset[sample_idx, t_idx + 1]
                dt = t_values[t_idx + 1] - t_values[t_idx]

                if mask_subset is not None:
                    m_curr = mask_subset[sample_idx, t_idx]
                    m_next = mask_subset[sample_idx, t_idx + 1]
                    valid = m_curr & m_next
                    if not valid.any():
                        continue
                    u_curr = u_curr[valid]
                    u_next = u_next[valid]

                residual = u_next - u_curr
                derivative = residual / dt

                residuals.append(residual)
                derivatives.append(derivative)

        if residuals:
            residuals = np.concatenate(residuals, axis=0)
            res_mean = np.mean(residuals, axis=0)
            res_std = np.std(residuals, axis=0) + EPSILON
            stats["res"] = {"mean": res_mean, "std": res_std}

            derivatives = np.concatenate(derivatives, axis=0)
            der_mean = np.mean(derivatives, axis=0)
            der_std = np.std(derivatives, axis=0) + EPSILON
            stats["der"] = {"mean": der_mean, "std": der_std}

        return stats

    def _convert_to_tensors_with_mask(
        self,
        u_train,
        u_val,
        u_test,
        c_train,
        c_val,
        c_test,
        x_train,
        x_val,
        x_test,
        mask_train,
        mask_val,
        mask_test,
        is_variable_coords,
    ) -> Dict:
        u_train = torch.tensor(u_train, dtype=self.dtype)
        u_val = torch.tensor(u_val, dtype=self.dtype)
        u_test = torch.tensor(u_test, dtype=self.dtype)

        if c_train is not None:
            c_train = torch.tensor(c_train, dtype=self.dtype)
            c_val = torch.tensor(c_val, dtype=self.dtype)
            c_test = torch.tensor(c_test, dtype=self.dtype)

        if is_variable_coords:
            x_train = torch.tensor(x_train, dtype=self.dtype)
            x_val = torch.tensor(x_val, dtype=self.dtype)
            x_test = torch.tensor(x_test, dtype=self.dtype)
        else:
            x_coord = torch.tensor(x_train, dtype=self.dtype)
            x_train = x_val = x_test = x_coord

        if mask_train is not None:
            mask_train = torch.tensor(mask_train, dtype=torch.bool)
            mask_val = torch.tensor(mask_val, dtype=torch.bool)
            mask_test = torch.tensor(mask_test, dtype=torch.bool)

        return {
            "train": {"c": c_train, "u": u_train, "x": x_train, "mask": mask_train},
            "val": {"c": c_val, "u": u_val, "x": x_val, "mask": mask_val},
            "test": {"c": c_test, "u": u_test, "x": x_test, "mask": mask_test},
        }

    def create_sequential_data_loaders(
        self, data_splits: Dict, is_variable_coords: bool, **kwargs
    ) -> Dict[str, Optional[DataLoader]]:
        from .data_utils import DynamicPairDatasetWithMask, collate_sequential_batch

        train_data = data_splits["train"]
        val_data = data_splits["val"]
        test_data = data_splits["test"]

        loaders = {}

        if getattr(self.dataset_config, "train", True):
            train_dataset = DynamicPairDatasetWithMask(
                u_data=train_data["u"],
                c_data=train_data["c"],
                x_data=train_data["x"] if is_variable_coords else None,
                mask_data=train_data["mask"],
                t_values=train_data["t"],
                metadata=self.metadata,
                max_time_diff=self.max_time_diff,
                stepper_mode=self.stepper_mode,
                stats=self.stats,
                use_time_norm=self.use_time_norm,
                is_variable_coords=is_variable_coords,
            )

            val_dataset = DynamicPairDatasetWithMask(
                u_data=val_data["u"],
                c_data=val_data["c"],
                x_data=val_data["x"] if is_variable_coords else None,
                mask_data=val_data["mask"],
                t_values=val_data["t"],
                metadata=self.metadata,
                max_time_diff=self.max_time_diff,
                stepper_mode=self.stepper_mode,
                stats=self.stats,
                use_time_norm=self.use_time_norm,
                is_variable_coords=is_variable_coords,
            )

            loaders["train"] = DataLoader(
                train_dataset,
                batch_size=self.dataset_config.batch_size,
                shuffle=self.dataset_config.shuffle,
                num_workers=self.dataset_config.num_workers,
                pin_memory=True,
                collate_fn=collate_sequential_batch,
            )

            loaders["val"] = DataLoader(
                val_dataset,
                batch_size=self.dataset_config.batch_size,
                shuffle=False,
                num_workers=self.dataset_config.num_workers,
                pin_memory=True,
                collate_fn=collate_sequential_batch,
            )
        else:
            loaders["train"] = None
            loaders["val"] = None

        test_dataset = DynamicPairDatasetWithMask(
            u_data=test_data["u"],
            c_data=test_data["c"],
            x_data=test_data["x"] if is_variable_coords else None,
            mask_data=test_data["mask"],
            t_values=test_data["t"],
            metadata=self.metadata,
            max_time_diff=self.max_time_diff,
            stepper_mode=self.stepper_mode,
            stats=self.stats,
            use_time_norm=self.use_time_norm,
            is_variable_coords=is_variable_coords,
        )

        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=self.dataset_config.batch_size,
            shuffle=False,
            num_workers=self.dataset_config.num_workers,
            pin_memory=True,
            collate_fn=collate_sequential_batch,
        )

        return loaders
