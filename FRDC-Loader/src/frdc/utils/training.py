from __future__ import annotations
from pathlib import Path

import lightning as pl
import numpy as np
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from frdc.load.dataset import FRDCDataset


def get_latest_ckpt_path(search_dir: Path, extention: str = "ckpt"):
    # This fetches all possible checkpoints and gets the latest one
    return sorted(
        search_dir.glob(f"**/*.{extention}"),
        key=lambda x: x.stat().st_mtime_ns,
    )[-1]


def plot_confusion_matrix(
    y_trues, y_preds, labels
) -> tuple[plt.Figure, plt.Axes]:
    # Plot the confusion matrix
    cm = confusion_matrix(y_trues, y_preds)

    fig, ax = plt.subplots(figsize=(10, 10))

    heatmap(
        cm,
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )

    fig.tight_layout(pad=3)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    return fig, ax


def predict(
    ds: FRDCDataset,
    model_cls: type[pl.LightningModule],
    ckpt_pth: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    m = model_cls.load_from_checkpoint(ckpt_pth)
    # Make predictions
    trainer = pl.Trainer(logger=False)
    pred = trainer.predict(m, dataloaders=DataLoader(ds, batch_size=32))

    y_preds = []
    y_trues = []
    for y_true, y_pred in pred:
        y_preds.append(y_pred)
        y_trues.append(y_true)
    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)
    return y_trues, y_preds


def accuracy(y_trues, y_preds) -> float:
    return (y_trues == y_preds).mean()
