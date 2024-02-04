from frdc.load.preset import FRDCDatasetPreset
from frdc.preprocess.extract_segments import extract_segments_from_bounds

ds = FRDCDatasetPreset.chestnut_20210510_43m()
ar, order = ds.get_ar_bands()
bounds, labels = ds.get_bounds_and_labels()
segments = extract_segments_from_bounds(ar, bounds)

# count number of examples for each label, filter only labels >= 2
counter = {}
for label in labels:
    counter[label] = counter.get(label, 0) + 1
filtered_labels = {label for label, count in counter.items() if count > 1}

import os
import numpy as np

label_dir = "chestnut_20210510_43m"
idxes = {}
for label, segment in zip(labels, segments):
    if counter[label] == 1:
        continue
    np.save(os.path.join(label_dir, f"{label}_{idxes.get(label, 0)}"), segment)
    idxes[label] = idxes.get(label, 0) + 1



