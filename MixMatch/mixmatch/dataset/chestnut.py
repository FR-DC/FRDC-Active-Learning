from __future__ import annotations
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import os
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.v2 import (
    RandomHorizontalFlip,
    RandomCrop,
)

tf_preproc = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

tf_aug = transforms.Compose(
    [
        lambda x: torch.nn.functional.pad(
            x,
            (
                4,
                4,
                4,
                4,
            ),
            mode="reflect",
        ),
        RandomCrop(32),
        RandomHorizontalFlip(),
    ]
)


class ChestnutDataset(Dataset):
    """
    Generic dataset class to read np arrays in from folder
    And create appropriate labels mappings
    """

    def __init__(self, 
                 root: str,
                 transform: Callable | None = None,
                 target_transform: Callable | None = None,
    ) -> None:
        self.species_map = os.listdir(root)
        self.data, self.targets = [], []
        self.transform = transform
        self.target_transform = target_transform
        for i, specie in enumerate(self.species_map):
            specie_path = os.path.join(root, specie)
            specie_list = []
            for fname in os.listdir(specie_path):
                specie_img = np.load(os.path.join(specie_path, fname))
                specie_list.append(specie_img)
            self.data.extend(specie_list)
            self.targets.extend([i] * len(specie_list))
        self.data = np.stack(self.data)
        self.targets = np.asarray(self.targets)
        
    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, target = self.data[index], self.targets[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target
    
    def standardize(self, 
                    stats: Tuple[np.ndarray, np.ndarray] = None) -> None:
        """
        Performs standardization based on dataset's mean and std
        """
        if stats:
            self.mean, self.std = stats
        else:
            self.calculate_stats()
        self.data -= self.mean
        self.data /= self.std
    
    def calculate_stats(self) -> None:
        """
        Calculates the mean and standard deviation from the dataset
        """
        self.mean = np.expand_dims(np.mean(self.data, axis=(0,1,2)), axis=(0,1,2))
        self.std = np.expand_dims(np.std(self.data, axis=(0,1,2)), axis=(0,1,2))

    # def replace_targets(self, labels: List[int] = None) -> None:
    #     if labels:
    #         for i, label in enumerate(labels):
    #             self.targets[self.targets == label] = i
    

class ChestnutSubset(ChestnutDataset):
    """
    Get subset of chestnut dataset
    """
    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        idxs: Sequence[int] | None = None,
        stats: Tuple[np.ndarray, np.ndarray] = None,
        labels: List[int] = None,
    ):
        super().__init__(root=root, 
                         transform=transform, 
                         target_transform=target_transform)
        if idxs is not None:
            self.data = self.data[idxs, ...]
            self.targets = self.targets[idxs]
        self.standardize(stats=stats)
        # self.replace_targets(labels=labels)


class ChestnutSubsetKAug(ChestnutDataset):
    """
    Create k augmentations for each image
    """
    def __init__(
        self,
        root: str,
        k_augs: int,
        aug: Callable,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        idxs: Sequence[int] | None = None,
        stats: Tuple[np.ndarray, np.ndarray] = None,
        labels: List[int] = None,
    ):
        super().__init__(root=root, 
                         transform=transform, 
                         target_transform=target_transform)
        self.k_augs = k_augs
        self.aug = aug
        if idxs is not None:
            self.data = self.data[idxs, ...]
            self.targets = self.targets[idxs]
        self.standardize(stats=stats)
        # self.replace_targets(labels=labels)

    def __getitem__(self, item):
        img, target = super().__getitem__(item)
        return tuple(self.aug(img) for _ in range(self.k_augs)), target


def get_dataloaders(
    train_dataset_dir: Path | str,
    test_dataset_dir: Path | str = None,
    train_lbl_size: float = 0.005,
    train_unl_size: float = 0.980,
    batch_size: int = 48,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, list[str]]:
    """Get the dataloaders for the CIFAR10 dataset.

    Notes:
        The train_lbl_size and train_unl_size must sum to less than 1.
        The leftover data is used for the validation set.

    Args:
        dataset_dir: The directory where the dataset is stored.
        train_lbl_size: The size of the labelled training set.
        train_unl_size: The size of the unlabelled training set.
        batch_size: The batch size.
        num_workers: The number of workers for the dataloaders.
        seed: The seed for the random number generators.

    Returns:
        4 DataLoaders: train_lbl_dl, train_unl_dl, val_unl_dl, test_dl
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    src_train_ds = ChestnutDataset(
        root=train_dataset_dir,
        transform=tf_preproc
    )

    train_size = len(src_train_ds)
    train_unl_size = int(train_size * train_unl_size)
    train_lbl_size = int(train_size * train_lbl_size)
    val_size = int(train_size - train_unl_size - train_lbl_size)

    targets = np.array(src_train_ds.targets)
    ixs = np.arange(len(targets))

    train_unl_ixs, lbl_ixs = train_test_split(
        ixs,
        train_size=train_unl_size,
        stratify=targets,
    )
    lbl_targets = targets[lbl_ixs]

    val_ixs, train_lbl_ixs = train_test_split(
        lbl_ixs,
        train_size=val_size,
        stratify=lbl_targets,
    )

    # create balanced idxes for each class
    # train_lbl_ixs, train_unl_ixs, val_ixs = [], [], []
    # for idx in range(len(os.listdir(train_dataset_dir))):
    #     indices = np.where(targets == idx)[0]
    #     np.random.shuffle(indices)
    #     train_lbl_ixs.extend(indices[:12])
    #     val_ixs.extend(indices[12:13])
        # train_unl_ixs.extend(indices[13:])

    train_lbl_ds = ChestnutSubsetKAug(
        root=train_dataset_dir,
        transform=tf_preproc,
        idxs=train_lbl_ixs,
        k_augs=1,
        aug=tf_aug,
    )
    train_unl_ds = ChestnutSubsetKAug(
        root=train_dataset_dir,
        transform=tf_preproc,
        idxs=train_unl_ixs,
        k_augs=2,
        aug=tf_aug,
        stats=(train_lbl_ds.mean, train_lbl_ds.std),
    )
    val_ds = ChestnutSubset(
        root=train_dataset_dir,
        transform=tf_preproc,
        idxs=val_ixs,
        stats=(train_lbl_ds.mean, train_lbl_ds.std),
    )

    dl_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    train_lbl_dl = DataLoader(
        train_lbl_ds, shuffle=True, drop_last=True, **dl_args
    )
    train_unl_dl = DataLoader(
        train_unl_ds, shuffle=True, drop_last=True, **dl_args
    )
    val_dl = DataLoader(val_ds, shuffle=False, **dl_args)
    if test_dataset_dir:
    #     test_ixes = []
    #     for idx in range(len(os.listdir(train_dataset_dir))):
    #         indices = np.where(targets == idx)[0]
    #         np.random.shuffle(indices)
    #         test_ixes.extend(indices[:min(20, len(indices))])
        test_ds = ChestnutSubset(
            root=test_dataset_dir,
            transform=tf_preproc,
            # idxs=test_ixes,
            stats=(train_lbl_ds.mean, train_lbl_ds.std),
        )
        print(f"Mean: {train_lbl_ds.mean}")
        print(f"Std: {train_lbl_ds.std}")

        test_dl = DataLoader(test_ds, shuffle=False, **dl_args)
    else:
        test_dl = None

    print(f"Dataset sizes - Train labeled: {len(train_lbl_ds)}, Train unlabeled: {len(train_unl_ds)}, Valid: {len(val_ds)}, Test: {len(test_ds)}")

    return (
        train_lbl_dl,
        train_unl_dl,
        val_dl,
        test_dl,
        src_train_ds.species_map,
    )
