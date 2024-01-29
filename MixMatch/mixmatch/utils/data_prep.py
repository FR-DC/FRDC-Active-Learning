from collections import defaultdict
from typing import List
from pathlib import Path
import os
import numpy as np

def process_segments_labels(segments: List[np.ndarray], 
                            labels: List[str],
                            home_path: Path | str,
                            grid_sz: int,
                            mode: str = "all" | "rgb"):
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
                                 size: int = 48):
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