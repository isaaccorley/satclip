import os
from typing import Any, Callable, Dict, Optional

import pandas as pd
import rasterio
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset
import matplotlib.pyplot as plt
import numpy as np
import torch

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Sampler

from .transforms import get_pretrained_s2_train_transform, get_s2_train_transform, get_uint16_train_transform, coordinate_jitter

CHECK_MIN_FILESIZE = 10000 # 10kb


class SpatialBatchSampler(Sampler):
    """Mixed spatial batches: each batch is many small local groups whose anchors
    are spread globally at random.

    A batch is built from batch_size//group_size groups; each group is a random
    anchor + its (group_size-1) nearest unused neighbours. So the batch keeps the
    normal globally-diverse contrastive task (random anchors) AND contains genuine
    near pairs (the neighbours) for the soft loss to act on. group_size=2 -> random
    anchor+nearest-neighbour pairs; group_size=batch_size -> one pure local cluster
    (the old neighbours-only behaviour, which lacks global diversity). Reshuffled
    each epoch.
    """

    def __init__(self, coords_lonlat, batch_size, group_size=None, shuffle=True, seed=0):
        from sklearn.neighbors import BallTree
        self.batch_size = batch_size
        self.group_size = int(group_size) if group_size else batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.n = len(coords_lonlat)
        self.latlon = np.deg2rad(np.asarray(coords_lonlat, dtype=np.float64)[:, [1, 0]])
        self.tree = BallTree(self.latlon, metric="haversine")
        self._epoch = 0

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        used = np.zeros(self.n, dtype=bool)
        order = rng.permutation(self.n) if self.shuffle else np.arange(self.n)
        gs = self.group_size
        k = int(min(self.n, gs * 8))  # query enough that gs are still unused
        batch = []
        for anchor in order:
            if used[anchor]:
                continue
            if gs == 1:
                grp = [int(anchor)]
            else:
                _, nbr = self.tree.query(self.latlon[anchor : anchor + 1], k=k)
                grp = [int(i) for i in nbr[0] if not used[i]][:gs]
            for i in grp:
                used[i] = True
            batch.extend(grp)
            if len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]
        if batch:
            yield batch

class CachedEmbedDataset(torch.utils.data.Dataset):
    """Precomputed frozen-backbone pre-head image features + coords (no image I/O).

    Built once by scripts/cache_image_embeddings.py from the frozen MoCo ViT-S/16
    (center crop, no augmentation -- valid because SatCLIP pretraining uses none).
    Returns {"image": pre-head feature [F], "point": (lon,lat)}; the model's
    encode_image applies only the trainable head to the 2-D feature. Coordinate
    jitter is kept (a location aug, cheap). Exposes .points for SpatialBatchSampler.
    """

    def __init__(self, emb_path):
        d = np.load(emb_path)
        self.Z = d["Z"].astype(np.float32)        # [N, F] frozen pre-head features
        coords = d["coords"].astype(np.float64)   # [N, 2] (lon, lat)
        self.points = [tuple(c) for c in coords]
        print(f"loaded {len(self.Z)} cached image embeddings dim={self.Z.shape[1]} from {emb_path}")

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, i):
        point = coordinate_jitter(torch.tensor(self.points[i]))
        return {"image": torch.from_numpy(self.Z[i]), "point": point}


class S2GeoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/geoclip_s2",
        batch_size: int = 64,
        num_workers: int = 6,
        crop_size: int = 256,
        val_random_split_fraction: float = 0.1,
        transform: str = 'pretrained',
        mode: str = "both",
        pin_memory: bool = False,
        spatial_batching: bool = False,
        spatial_group_size: int = None,
        cached_emb_path: str = None,
    ):
        super().__init__()
        self.cached_emb_path = cached_emb_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.spatial_batching = spatial_batching
        self.spatial_group_size = spatial_group_size
        if transform=='pretrained':
            self.train_transform = get_pretrained_s2_train_transform(resize_crop_size=crop_size)
        elif transform=='default':
            self.train_transform = get_s2_train_transform()
        elif transform=='gpu':
            # workers hand off raw uint16; float cast + B10 + augmentation happen on
            # the GPU in SatCLIPLightningModule.on_after_batch_transfer
            self.train_transform = get_uint16_train_transform()
        else:
            self.train_transform = transform

        self.val_random_split_fraction = val_random_split_fraction
        self.mode = mode
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_dir):
            print("""
            No dataset found. To download, please follow instructions on: https://github.com/microsoft/satclip
            """)

    def setup(self, stage="fit"):
        if self.cached_emb_path is not None:
            # train on precomputed frozen-backbone pre-head image features (no image I/O)
            dataset = CachedEmbedDataset(self.cached_emb_path)
        else:
            dataset = S2Geo(root=self.data_dir, transform=self.train_transform, mode=self.mode)

        N_val = int(len(dataset) * self.val_random_split_fraction)
        N_train = len(dataset) - N_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [N_train, N_val])

    def _coords_for(self, subset):
        # subset-local coords aligned to the Subset's positional indices
        full = subset.dataset
        return np.asarray(full.points, dtype=np.float64)[np.asarray(subset.indices)]

    def _make_loader(self, dataset, shuffle):
        kw = dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
        if self.spatial_batching:
            sampler = SpatialBatchSampler(
                self._coords_for(dataset), self.batch_size,
                group_size=self.spatial_group_size, shuffle=shuffle, seed=0,
            )
            return DataLoader(dataset, batch_sampler=sampler, **kw)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, **kw)

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        raise NotImplementedError

class S2Geo(NonGeoDataset):
    """S2-100K dataset.

    This dataset contains 100,000 256x256 patches of 12 band Sentinel imagery sampled randomly
    from Sentinel 2 scenes on the Microsoft Planetary Computer that have <20% cloud cover,
    intersect land, and were captured between 2021-01-01 and 2023-05-17 (there are 2,359,972
    such scenes).
    """

    # Presence of the index and image dir is enough; specific patch indices are
    # mirror-specific (the davanstrien/satclip HF mirror ships ~94,164 of the
    # 100,000 patches index.csv references, and is missing patch_99999.tif).
    validation_filenames = [
        "index.csv",
        "images/",
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        mode: Optional[str] = "both",
    ) -> None:
        """Initialize a new S2-100K dataset instance.
        Args:
            root: root directory of S2-100K pre-sampled dataset
            transform: torch transform to apply to a sample
            mode: which data to return (options are "both" or "points"), useful for embedding locations without loading images 
        """
        assert mode in ["both", "points"]
        self.root = root
        self.transform = transform
        self.mode = mode
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        index_fn = "index.csv"

        df = pd.read_csv(os.path.join(self.root, index_fn))
        self.filenames = []
        self.points = []

        n_skipped_files = 0
        n_missing_files = 0
        for i in range(df.shape[0]):
            filename = os.path.join(self.root, "images", df.iloc[i]["fn"])

            # index.csv may reference patches absent from this mirror; skip them
            # (deterministic given a fixed index, so every run sees the same set).
            if not os.path.exists(filename):
                n_missing_files += 1
                continue

            if os.path.getsize(filename) < CHECK_MIN_FILESIZE:
                n_skipped_files += 1
                continue

            self.filenames.append(filename)
            self.points.append(
                (df.iloc[i]["lon"], df.iloc[i]["lat"])
            )

        print(f"skipped {n_skipped_files}/{len(df)} images because they were smaller "
              f"than {CHECK_MIN_FILESIZE} bytes... they probably contained nodata pixels")
        print(f"skipped {n_missing_files}/{len(df)} images missing from this dataset mirror")
        print(f"using {len(self.filenames)} images")

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            dictionary with "image" and "point" keys where point is in (lon, lat) format
        """
        point = torch.tensor(self.points[index])
        sample = {"point": point}

        if self.mode == "both":
            with rasterio.open(self.filenames[index]) as f:
                data = f.read()  # raw uint16; the transform decides the dtype
            sample["image"] = data
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.
        Returns:
            length of dataset
        """
        return len(self.filenames)

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.
        Returns:
            True if the dataset directories and split files are found, else False
        """
        
        for filename in self.validation_filenames:
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                print(filepath +' missing' )
                return False
        return True

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.
        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        ncols = 1

        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        ax.imshow(image[:, :, [3,2,1]] / 4000)
        ax.axis("off")

        if show_titles:
            ax.set_title(f"({sample['point'][0]:0.4f}, {sample['point'][1]:0.4f})")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig