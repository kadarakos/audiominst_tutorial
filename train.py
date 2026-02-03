import sys

import torch

from statistics import mean
from string import Template
from pathlib import Path

from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits


CKPT_FNAME = Template("$name-$epoch.pt")


def save_model(
    model: nn.Module,
    optimizer: Optimizer,
    loss: float,
    ckpt_dir: Path,
    epoch: int,
    name: str = "",
) -> None:
    fname = CKPT_FNAME.substitute(name=name, epoch=epoch)
    path = ckpt_dir / fname
    print(f"Saving model: {path}")
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(state_dict, path)


def evaluate(model: nn.Module, eval_loader: DataLoader) -> float:
    accs = []
    for batch in tqdm(eval_loader, desc="Evaluating"):
        wavs, labels = batch
        preds = model.predict(wavs)
        acc = float((preds == labels).sum()) / len(labels)
        accs.append(acc)
    return mean(accs)


def training_loop(
    model: nn.Module,
    optimizer: Optimizer,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    ckpt_dir: Path,
    name: str = "",
    tolerance: int = 2,
):
    ckpt_dir = Path(ckpt_dir)
    best_acc = evaluate(model, test_loader)
    print(f"Accuracy before training: {best_acc}")
    avg_loss = 0
    total_steps = 0
    noimprove = 0
    for epoch in range(epochs):
        desc = f"Epoch: {epoch}"
        pbar = tqdm(train_loader, desc=desc)
        for batch in pbar:
            total_steps += 1
            optimizer.zero_grad()
            wav, labels = batch
            logits = model(wav).squeeze()
            loss = binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            avg_loss += (loss.item() - avg_loss) / total_steps
            pbar.set_postfix_str(f"Loss: {round(avg_loss, 3)}")
        acc = evaluate(model, test_loader)
        if acc > best_acc:
            best_acc = acc
            print(f"New best found: {best_acc}")
            save_model(
                model=model,
                optimizer=optimizer,
                loss=loss.item(),
                epoch=epoch,
                ckpt_dir=ckpt_dir,
                name=name,
            )
        else:
            if noimprove == tolerance:
                print(f"Tolerance of {tolerance} reached, quitting.")
                sys.exit()
            noimprove += 1
