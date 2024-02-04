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

from dataset.chestnut import get_dataloaders
import models.wideresnet as models
from utils.ema import WeightEMA
from utils.eval import validate, train, confusion_mat
from utils.loss import SemiLoss
from utils.checkpoint import mkdir_p, save_checkpoint
from tensorboardX import SummaryWriter


def main(
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    train_iteration: int = 256,
    ema_decay: float = 0.999,
    lambda_u: float = 75,
    alpha: float = 0.75,
    t: float = 0.5,
    device: str = "cuda",
    seed: int = 42,
    out: Path | str = "../../Chestnut_logs",
):
    if not os.path.isdir(out):
        mkdir_p(out)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    best_acc = 0

    # Data
    print(f"==> Preparing chestnut data")

    (
        train_lbl_dl,
        train_unl_dl,
        val_loader,
        test_loader,
        classes,
    ) = get_dataloaders(
        train_dataset_dir="../../chestnut_20201218_48",
        test_dataset_dir="../../chestnut_20210510_43m_48",
        train_lbl_size=0.2,
        train_unl_size=0.6,
        batch_size=batch_size,
        seed=seed,
    )
    
    # Model
    print("==> creating WRN-28-2")

    model = models.WideResNet(num_classes=9).to(device)
    ema_model = deepcopy(model).to(device)
    for param in ema_model.parameters():
        param.detach_()

    # cudnn.benchmark = True
    print(
        "    Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    )

    train_loss_fn = SemiLoss()
    val_loss_fn = nn.CrossEntropyLoss()
    train_optim = optim.Adam(model.parameters(), lr=lr)

    ema_optim = WeightEMA(model, ema_model, alpha=ema_decay, lr=lr)

    test_accs = []
    writer = SummaryWriter(out)
    step = 0
    # Train and val
    for epoch in range(epochs):
        print("\nEpoch: [%d | %d] LR: %f" % (epoch + 1, epochs, lr))

        train_loss, train_lbl_loss, train_unl_loss = train(
            train_lbl_dl=train_lbl_dl,
            train_unl_dl=train_unl_dl,
            model=model,
            optim=train_optim,
            ema_optim=ema_optim,
            loss_fn=train_loss_fn,
            epoch=epoch,
            device="cuda",
            train_iters=train_iteration,
            lambda_u=lambda_u,
            mix_beta_alpha=alpha,
            epochs=epochs,
            sharpen_temp=t,
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

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step)

        # save model
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
                checkpoint=out)

        print(
            f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f} | "
            f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f} | "
            f"Best Acc: {best_acc:.3f} | "
            f"Mean Acc: {np.mean(test_accs[-20:]):.3f} | "
            f"LR: {lr:.5f} | "
            f"Train Loss X: {train_lbl_loss:.3f} | "
            f"Train Loss U: {train_unl_loss:.3f} "
        )

    writer.close()

    resume = os.path.join(out, "model_best.pth.tar")
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    confusion_mat(test_dl=test_loader, out=out, species_map=classes, model=model, device=device)

    print("Best acc:")
    print(best_acc)

    print("Mean acc:")
    print(np.mean(test_accs[-20:]))

    return best_acc, np.mean(test_accs[-20:])

if __name__ == "__main__":
    main()