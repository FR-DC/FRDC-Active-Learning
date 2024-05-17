import random
from copy import deepcopy

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import StepLR

from dataset.al_ds import get_datasets
import models.wideresnet as models
from utils.sampling import confidence_sampling, margin_sampling, entropy_sampling, random_sampling, cluster_sampling, model_outlier_sampling, hybrid_sampling
from utils.ema import WeightEMA
from utils.eval import validate, train_sl, visualize_pca
# from utils.loss import SemiLoss
from utils.checkpoint import mkdir_p, save_checkpoint
from typing import Callable
from tensorboardX import SummaryWriter

def main(batch_size: int = 128,
         query: Callable = hybrid_sampling,
         train_iteration: int = 16,
         n_queries: int = 10,
         n_instances: int = 20,
         alpha: float = 0.75,
         lr: float = 0.00003,
         epochs: int = 5,
         ema_decay: float = 0.999,
         device: str = "cuda",
         saved_model_path: Path | str = "../../Chestnut_logs/48_ssl_8bands/model_best_48_ssl",
         bestname: Path | str = "model_best_al",
         seed: int = 42,
         out: Path | str = "../../Active-Learning_logs/hybrid"):
    
    if not os.path.isdir(out):
        mkdir_p(out)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data
    print(f"==> Preparing chestnut data")

    (
        train_lbl_ds,
        train_unl_ds,
        val_ds,
        val_loader,
        test_loader,
        classes,
    ) = get_datasets(
        train_dataset_dir="../../chestnut_20201218_48_remote",
        test_dataset_dir="../../chestnut_20210510_43m_48_remote",
        train_lbl_size=0.2,
        train_unl_size=0.6,
        batch_size=batch_size,
        seed=seed,
    )
    
    # Model
    print("==> creating WRN-28-2")

    model = models.WideResNet(num_classes=9).to(device)
    if saved_model_path:
        checkpoint = torch.load(saved_model_path)
        model.load_state_dict(checkpoint['ema_state_dict'])
        models.freeze_all_layers(model)
    ema_model = deepcopy(model).to(device)
    for param in ema_model.parameters():
        param.detach_()

    print(
        "    Total params: %.3f * 10^3"
        % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000.0)
    )

    dl_args = dict(
        batch_size=batch_size,
        num_workers=0,
    )

    train_loss_fn = nn.CrossEntropyLoss()
    val_loss_fn = nn.CrossEntropyLoss()
    train_optim = optim.Adam(model.parameters(), lr=lr)
    
    ema_optim = WeightEMA(model, ema_model, alpha=ema_decay, lr=lr)

    test_accs = []
    writer = SummaryWriter(out)

    max_uncertainties, mean_accs = [], []

    # the active learning loop
    for idx in range(n_queries):
        print('Query no. %d' % (idx + 1))
        unl_data = train_unl_ds.to_tensor().to(device)
        query_preds, query_idx, query_instance, max_uncertainty = query(ema_model, val_ds.to_tensor().to(device), unl_data, n_instances=n_instances)
        if idx % 3 == 0:
            visualize_pca(idxes = query_idx, 
                          preds = query_preds,
                          targets = train_unl_ds.targets,
                          title = "Hybrid sampling",
                          out = out,
                          classes = classes,
                          iteration = idx)
        max_uncertainties.append(max_uncertainty)
        # add queried instances to labelled dataset
        train_lbl_ds.concat(new_samples=query_instance, new_labels=train_unl_ds.targets[query_idx])
        # remove queried instance from unlabelled dataset
        train_unl_ds.remove(query_idx)
        train_unl_ds.standardize(stats=(train_lbl_ds.mean, train_lbl_ds.std))

        # convert datasets to dataloaders
        train_lbl_dl = DataLoader(
            train_lbl_ds, shuffle=True, drop_last=True, **dl_args
        )
        best_acc = 0.0
        mean_acc = []
        for epoch in range(epochs):
            if epoch < int(0.5 * epochs):
                models.freeze_all_layers(model)
            else:
                models.unfreeze_all_layers(model)
            # run training
            train_loss = train_sl(
                train_lbl_dl=train_lbl_dl,
                model=model,
                optim=train_optim,
                ema_optim=ema_optim,
                loss_fn=train_loss_fn,
                device="cuda",
                train_iters=train_iteration,
                mix_beta_alpha=alpha,
            )

            def val_ema(dl: DataLoader):
                return validate(
                    valloader=dl,
                    model=ema_model,
                    loss_fn=val_loss_fn,
                    device=device,
                )

            _, train_acc = val_ema(train_lbl_dl)
            val_loss, val_acc = val_ema(val_loader)
            if test_loader:
                test_loss, test_acc = val_ema(test_loader)
                test_accs.append(test_acc)
            else:
                test_loss, test_acc = 0.0, 0.0

            step = train_iteration * (epoch + 1)

            # save model
            mean_acc.append(test_acc)
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_checkpoint(state={
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'acc': val_acc,
                    'best_acc': best_acc,
                    'optimizer' : train_optim.state_dict()}, 
                    is_best=is_best,
                    bestname=f"{bestname}_{epoch}.pth.tar",
                    checkpoint=out)

            print(
                f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f} | "
                f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f} | "
                f"Best Acc: {best_acc:.3f} | "
                f"Mean Acc: {np.mean(test_accs[-20:]):.3f} | "
                f"LR: {lr:.5f} | "
            )

        print("Best acc:")
        print(best_acc)
        print(f"Mean acc: {mean_acc}")
        mean_accs.append(np.mean(mean_acc))
        writer.add_scalar("mean_accuracy/test_accuracy", mean_accs[-1], (idx+1)*20)
        writer.add_scalar("uncertainty/maximum_uncertainty", max_uncertainties[-1], (idx+1)*20)
        writer.close()

        print("Mean acc:")
        print(np.mean(test_accs[-20:]))
    
    print(f"Max accs: {mean_accs}")
    print(f"Max uncertainties: {max_uncertainties}")
if __name__ == "__main__":
    main()