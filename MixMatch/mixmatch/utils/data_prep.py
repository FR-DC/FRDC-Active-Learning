from collections import defaultdict
from typing import List, Callable, Dict, Tuple
from pathlib import Path
import os
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


def reconstruct_image_from_cells(test_dataset_dir: str,
                                 size: int = 48) -> Dict[str, Dict[int, np.ndarray]]:
    dims, cells = defaultdict(lambda: (-1, -1)), defaultdict(lambda: dict())
    spec_images = {}
    for species in os.listdir(test_dataset_dir):
        species_path = os.path.join(test_dataset_dir, species)
        for fname in os.listdir(species_path):
            _split_name = fname.split("_")
            idx, i, j = int(_split_name[1]), int(_split_name[2]), int(_split_name[3][:-4])
            cell = np.load(os.path.join(species_path, fname))
            cells[idx][(i, j)] = cell
            max_x, max_y = max(i, dims[idx][0]), max(j, dims[idx][1])
            dims[idx] = (max_x, max_y)

        imgs = {}
        for idx, (n_c, n_r) in dims.items():
            img_arr = np.zeros(((n_r+1) * size, (n_c+1) * size, 3), dtype=np.uint8)
            for (i, j), cell in cells[idx].items():
                img_arr[j*size:(j+1)*size, i*size:(i+1)*size, :] = cell
            img_arr = img_arr.astype(np.uint8)
            imgs[idx] = img_arr
        spec_images[species] = imgs
    return spec_images

def load_segments_labels(ds_path: Path | str) -> Tuple[List[np.ndarray], 
                                                       List[np.ndarray]]:
    segments, labels = [], []
    for species in os.listdir(ds_path):
        segments.append(np.load(os.path.join(ds_path, species)))
        labels.append(species.split("_")[0])
    return segments, labels

def main():
    base_dir = "../../"
    train_segments, train_labels = load_segments_labels(ds_path=os.path.join(base_dir, "chestnut_20201218"))
    test_segments, test_labels = load_segments_labels(ds_path=os.path.join(base_dir, "chestnut_20210510_43m"))
    process_segments_labels(segments=train_segments,
                            labels=train_labels,
                            home_path=os.path.join(base_dir, "chestnut_20201218_48"),
                            grid_sz=48,
                            mode="all")
    process_segments_labels(segments=test_segments,
                        labels=test_labels,
                        home_path=os.path.join(base_dir, "chestnut_20210510_43m_48"),
                        grid_sz=48,
                        mode="all")
    
if __name__ == "__main__":
    main()