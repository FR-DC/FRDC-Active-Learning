from typing import Callable, Sequence, List

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.functional import one_hot
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path

from utils.ema import WeightEMA
from utils.interleave import interleave
from utils.loss import SemiLoss
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from seaborn import heatmap


def mix_up(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = np.random.beta(alpha, alpha)
    ratio = max(ratio, 1 - ratio)

    shuf_idx = torch.randperm(x.size(0))

    x_mix = ratio * x + (1 - ratio) * x[shuf_idx]
    y_mix = ratio * y + (1 - ratio) * y[shuf_idx]
    return x_mix, y_mix


def guess_labels(
    model: nn.Module,
    x_unls: list[torch.Tensor],
    sharpen_temp: float,
) -> torch.Tensor:
    """Guess labels from the unlabelled data"""
    y_unls = [torch.softmax(model(u), dim=1) for u in x_unls]
    p = sum(y_unls) / 2
    pt = p ** (1 / sharpen_temp)
    return pt / pt.sum(dim=1, keepdim=True).detach()


def train(
    *,
    train_lbl_dl: DataLoader,
    train_unl_dl: DataLoader,
    model: nn.Module,
    optim: Optimizer,
    ema_optim: WeightEMA,
    loss_fn: SemiLoss,
    epoch: int,
    epochs: int,
    device: str,
    train_iters: int,
    lambda_u: float,
    mix_beta_alpha: float,
    sharpen_temp: float,
) -> tuple[float, float, float]:
    losses = []
    losses_x = []
    losses_u = []
    n = []

    lbl_iter = iter(train_lbl_dl)
    unl_iter = iter(train_unl_dl)

    model.train()
    for batch_idx in tqdm(range(train_iters)):
        try:
            (x_lbl,), y_lbl = next(lbl_iter)
        except StopIteration:
            lbl_iter = iter(train_lbl_dl)
            (x_lbl,), y_lbl = next(lbl_iter)

        try:
            x_unls, _ = next(unl_iter)
        except StopIteration:
            unl_iter = iter(train_unl_dl)
            x_unls, _ = next(unl_iter)

        batch_size = x_lbl.size(0)
        y_lbl = one_hot(y_lbl.long(), num_classes=9)

        x_lbl = x_lbl.to(device)
        y_lbl = y_lbl.to(device)
        x_unls = [u.to(device) for u in x_unls]

        with torch.no_grad():
            y_unl = guess_labels(
                model=model,
                x_unls=x_unls,
                sharpen_temp=sharpen_temp,
            )

        x = torch.cat([x_lbl, *x_unls], dim=0)
        y = torch.cat([y_lbl, y_unl, y_unl], dim=0)
        x_mix, y_mix = mix_up(x, y, mix_beta_alpha)

        # interleave labeled and unlabeled samples between batches to
        # get correct batchnorm calculation
        x_mix = list(torch.split(x_mix, batch_size))
        x_mix = interleave(x_mix, batch_size)

        y_mix_pred = [model(x) for x in x_mix]

        # put interleaved samples back
        y_mix_pred = interleave(y_mix_pred, batch_size)

        y_mix_lbl_pred = y_mix_pred[0]
        y_mix_lbl = y_mix[:batch_size]
        y_mix_unl_pred = torch.cat(y_mix_pred[1:], dim=0)
        y_mix_unl = y_mix[batch_size:]

        # TODO: Pretty ugly that we throw in epoch, epochs and lambda_u here
        loss_lbl, loss_unl, loss_unl_scale = loss_fn(
            x_lbl=y_mix_lbl_pred,
            y_lbl=y_mix_lbl,
            x_unl=y_mix_unl_pred,
            y_unl=y_mix_unl,
            epoch=epoch + batch_idx / train_iters,
            lambda_u=lambda_u,
            epochs=epochs,
        )

        loss = loss_lbl + loss_unl_scale * loss_unl

        losses.append(loss)
        losses_x.append(loss_lbl)
        losses_u.append(loss_unl)
        n.append(x_lbl.size(0))

        optim.zero_grad()
        loss.backward()
        optim.step()
        ema_optim.step()

    return (
        sum([loss * n for loss, n in zip(losses, n)]) / sum(n),
        sum([loss * n for loss, n in zip(losses_x, n)]) / sum(n),
        sum([loss * n for loss, n in zip(losses_u, n)]) / sum(n),
    )

def train_sl(
    *,
    train_lbl_dl: DataLoader,
    model: nn.Module,
    optim: Optimizer,
    ema_optim: WeightEMA,
    loss_fn: nn.CrossEntropyLoss,
    device: str,
    train_iters: int,
    mix_beta_alpha: float,
) -> float:
    losses = []
    n = []

    lbl_iter = iter(train_lbl_dl)

    model.train()
    for batch_idx in tqdm(range(train_iters)):
        try:
            (x_lbl,), y_lbl = next(lbl_iter)
        except StopIteration:
            lbl_iter = iter(train_lbl_dl)
            (x_lbl,), y_lbl = next(lbl_iter)

        y_lbl = one_hot(y_lbl.long(), num_classes=9)

        x_lbl = x_lbl.to(device)
        y_lbl = y_lbl.to(device)

        x_mix, y_mix = mix_up(x_lbl, y_lbl, mix_beta_alpha)

        y_mix_pred = model(x_mix)

        loss_lbl = loss_fn(
            y_mix_pred,
            y_mix
        )

        losses.append(loss_lbl)
        n.append(x_lbl.size(0))

        optim.zero_grad()
        loss_lbl.backward()
        optim.step()
        ema_optim.step()

    return sum([loss * n for loss, n in zip(losses, n)]) / sum(n)


def validate(
    *,
    valloader: DataLoader,
    model: nn.Module,
    loss_fn: Callable,
    device: str,
):
    n = []
    losses = []
    accs = []

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(valloader):
            # TODO: Pretty hacky but this is for the train loader.
            if isinstance(x, Sequence):
                x = x[0]

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y.long())

            # TODO: Technically, we shouldn't * 100, but it's fine for now as
            #  it doesn't impact training
            acc = (
                accuracy(
                    y_pred,
                    y,
                    task="multiclass",
                    num_classes=y_pred.shape[1],
                )
                * 100
            )
            losses.append(loss.item())
            accs.append(acc.item())
            n.append(x.size(0))

    # return weighted mean
    return (
        sum([loss * n for loss, n in zip(losses, n)]) / sum(n),
        sum([top * n for top, n in zip(accs, n)]) / sum(n),
    )


def confusion_mat(test_dl: DataLoader, 
                  species_map: dict,
                  model: nn.Module,
                  out: Path | str,
                  device: str):
    y_trues, y_preds = [], []
    logits = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test_dl):
            # TODO: Pretty hacky but this is for the train loader.
            if isinstance(x, Sequence):
                x = x[0]

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            preds = torch.argmax(y_pred, dim=1)
            y_trues.append(y)
            y_preds.append(preds)
            logits.append(y_pred)

    y_trues = torch.cat(y_trues)
    y_preds = torch.cat(y_preds)
    logits = torch.cat(logits)
    # sanity check for accuracy
    acc = (
        accuracy(
            logits,
            y_trues,
            task="multiclass",
            num_classes=logits.shape[1],
        )
        * 100
    )
    print(f"Best acc is: {acc}")
    # Plot the confusion matrix
    cm = confusion_matrix(y_trues.cpu().numpy(), y_preds.cpu().numpy())
    np.save(os.path.join(out, "confusion_matrix"), cm)

    plt.figure(figsize = (10,7))
    heatmap(cm, xticklabels=species_map, yticklabels=species_map, annot=True, fmt='g')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=30)
    plt.savefig(os.path.join(out, "heatmap.png"), dpi=400)
    return

def visualize_pca(idxes: np.ndarray, 
                  preds: np.ndarray,
                  targets: np.ndarray,
                  title: str,
                  out: Path | str,
                  classes: List[str],
                  iteration: int):

    pca = PCA(n_components=2)
    x_new = pca.fit_transform(preds)
    x, y = x_new.T

    x_max = x[idxes]
    y_max = y[idxes]

    fig, ax = plt.subplots()
    cdict = {0: 'cyan', 1: 'blue', 2: 'red', 3:'orange', 4: 'purple', 5: 'green', 6: 'yellow', 7: 'magenta', 8: 'brown', 9: 'pink'}
    for i, g in enumerate(classes):
        ix = np.where(targets == i)
        ax.scatter(x[ix], y[ix], c = cdict[i], label = g, s = 100, alpha=0.2)

    ax.scatter(x_max, y_max, c='black', s=100, alpha=0.7, marker='^')
    ax.legend(loc='upper left', prop={'size': 6})
    ax.set_title(title)
    fig.savefig(os.path.join(out, f"pca_{iteration}"), dpi=400)