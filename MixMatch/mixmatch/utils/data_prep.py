from collections import defaultdict
from typing import List, Callable, Dict, Tuple
from pathlib import Path
from pylab import *
import os
import torch
import torch.nn as nn
import numpy as np


def _fn_per_band(ar: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]):
    """Runs an operation for each band in an NDArray."""
    ar = ar.copy()
    ar_bands = []
    for band in range(ar.shape[-1]):
        ar_band = ar[:, :, band]
        ar_band = fn(ar_band)
        ar_bands.append(ar_band)

    return np.stack(ar_bands, axis=-1)


def scale_0_1_per_band(
    ar: np.ndarray, epsilon: float | bool = False
) -> np.ndarray:
    """Scales an NDArray from 0 to 1 for each band independently

    Args:
        ar: NDArray of shape (H, W, C), where C is the number of bands.
        epsilon: If True, then we add a small epsilon to the denominator to
            avoid division by zero. If False, then we do not add epsilon.
            If a float, then we add that float as epsilon.
    """
    epsilon = 1e-7 if epsilon is True else epsilon

    return _fn_per_band(
        ar,
        lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + epsilon),
    )


def process_segments_labels(segments: List[np.ndarray], 
                            labels: List[str],
                            home_path: Path | str,
                            grid_sz: int,
                            mode: str):
    # counter to find out how many images of each species there are
    label_dict = {}
    for label in labels:
        label_dict[label]  = label_dict.get(label, 0) + 1
    # only select species with at least 2 examples, since there is more data to work with
    segments_processed, labels_processed = [], []
    for i, segment in enumerate(segments):
        if label_dict[labels[i]] == 1:
            continue
        if mode == "rgb":
            segment = segment[..., [2, 1, 0]]
        segment = scale_0_1_per_band(segment)
        segments_processed.append((segment, segment.shape))
        labels_processed.append(labels[i])

    # segment image into cells
    segment_cells = []
    total_samples = 0
    for idx, label in enumerate(labels_processed):
        label_dir = os.path.join(home_path, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        img, (h, w, _) = segments_processed[idx]
        cells = []
        x, y = w // grid_sz, h // grid_sz
        total_samples += x * y
        for j in range(y):
            for i in range(x):
                cell = img[(j*grid_sz):((j+1)*grid_sz), (i*grid_sz):((i+1)*grid_sz), :]
                # save cells with appropriate naming to npz files
                np.save(os.path.join(label_dir, f"{label}_{idx}_{i}_{j}"), cell)
                cells.append(cell)
        segment_cells.append(cells)
    print(f"Total samples is: {total_samples}")


def create_mask(model: nn.Module,
                mean: np.ndarray,
                std: np.ndarray,
                transforms: Callable,
                cell: np.ndarray,
                device: str):
    cell = (cell - mean) / std
    cell = cell.squeeze(0)
    cell = transforms(cell)
    cell = torch.unsqueeze(cell, 0)
    cell = cell.to(device, dtype=torch.float)
    model.eval()
    with torch.no_grad():
        pred = model(cell)
        pred_cls = torch.argmax(pred)
    return torch.full((48, 48, 1), pred_cls)


def reconstruct_image_from_cells(model: nn.Module,
                                 test_dataset_dir: str,
                                 ds_stats : Tuple[np.ndarray, np.ndarray],
                                 transforms: Callable,
                                 size: int = 48) -> Dict[str, Dict[int, np.ndarray]]:
    """
    returns spec_images
    which is a dictionary mapping each species to list of tuples
    with each tuple containing the reconstructed image and its prediction mask
    """
    spec_images = {}
    mean, std = ds_stats
    for species in os.listdir(test_dataset_dir):
        dims, cells = defaultdict(lambda: (-1, -1)), defaultdict(lambda: dict())
        species_path = os.path.join(test_dataset_dir, species)
        # preprocessing
        # extracts dimensions to reconstruct original image 
        for fname in os.listdir(species_path):
            _split_name = fname.split("_")
            idx, i, j = int(_split_name[1]), int(_split_name[2]), int(_split_name[3][:-4])
            cell = np.load(os.path.join(species_path, fname))
            mask = create_mask(model=model, 
                               mean=mean, 
                               std=std, 
                               transforms=transforms,
                               cell=cell, 
                               device="cuda")
            if cell.shape[-1] == 8:
                cell = cell[..., [2, 1, 0]]
            cells[idx][(i, j)] = (cell, mask)
            max_x, max_y = max(i, dims[idx][0]), max(j, dims[idx][1])
            dims[idx] = (max_x, max_y)

        imgs = {}
        for idx, (n_r, n_c) in dims.items():
            img_arr = np.zeros(((n_c+1) * size, (n_r+1) * size, 3))
            mask_arr = np.zeros(((n_c+1) * size, (n_r+1) * size, 1))
            for (i, j), (cell, mask) in cells[idx].items():
                img_arr[j*size:(j+1)*size, i*size:(i+1)*size, :] = cell
                mask_arr[j*size:(j+1)*size, i*size:(i+1)*size, :] = mask
            imgs[idx] = (img_arr, mask_arr)
        spec_images[species] = imgs
    return spec_images


def load_segments_labels(ds_path: Path | str) -> Tuple[List[np.ndarray], 
                                                       List[np.ndarray]]:
    segments, labels = [], []
    for species in os.listdir(ds_path):
        segments.append(np.load(os.path.join(ds_path, species)))
        labels.append(species.split("_")[0])
    return segments, labels


def _counts(imgs: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
            species: str, index: int):
    unique, counts = np.unique(imgs[species][index][1], return_counts=True)
    return dict(zip(unique, counts))


def plot_visuals(imgs: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]], 
                 classes: List[str], species: str, index: int, cell_size: int = 48):

    img, mask = imgs[species][index]
    mask = mask.squeeze()

    # Grid lines at these intervals (in pixels)
    # dx and dy can be different
    dx, dy = cell_size, cell_size

    # Custom (rgb) grid color
    grid_color = [0,0,0]

    # Modify the image to include the grid
    img[:,::dy,:] = grid_color
    img[::dx,:,:] = grid_color

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img)
    ax[1].imshow(img)
    cmap = cm.get_cmap('jet', 9)    # 9 discrete colors
    # put those patched as legend-handles into the legend
    plot = ax[1].imshow(mask, cmap=cmap, interpolation='bilinear', vmax=8, vmin=0, alpha=0.7)
    cbar = fig.colorbar(plot, location="bottom")
    cbar.ax.set_xticklabels(classes, rotation = 45)  # vertically oriented colorbar
    fig.tight_layout()


def main():
    base_dir = "../../../"
    train_segments, train_labels = load_segments_labels(ds_path=os.path.join(base_dir, "chestnut_20201218_remote"))
    test_segments, test_labels = load_segments_labels(ds_path=os.path.join(base_dir, "chestnut_20210510_43m_remote"))
    process_segments_labels(segments=train_segments,
                            labels=train_labels,
                            home_path=os.path.join(base_dir, "chestnut_20201218_48_rgb_remote"),
                            grid_sz=48,
                            mode="rgb")
    process_segments_labels(segments=test_segments,
                        labels=test_labels,
                        home_path=os.path.join(base_dir, "chestnut_20210510_43m_48_rgb_remote"),
                        grid_sz=48,
                        mode="rgb")
    
if __name__ == "__main__":
    main()