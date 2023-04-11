import os

import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

from pytorch_lightning.loggers import MLFlowLogger

from trainer import LitTrainer


def argmax(a):
    return max(range(len(a)), key=lambda x: a[x])


def get_dataloaders(dataset, test_data=None):
    train_size = round(len(dataset) * 0.8)
    validate_size = len(dataset) - train_size
    train_data, validate_data = random_split(dataset, [train_size, validate_size])

    # For 8 CPU cores
    return DataLoader(train_data, num_workers=8), \
        DataLoader(validate_data, num_workers=8), \
        DataLoader(test_data, num_workers=8) if test_data else None


def train_loop(net, batch, loss_fn, optim, device="cuda"):
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    y_pred = net(x).reshape(1, -1)
    loss = loss_fn(y_pred, y)
    truth_count = argmax(y_pred.flatten()) == y

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.item(), truth_count


def train_net_manually(net, optim, loss_fn, train_loader, validate_loader=None, epochs=10, device="cuda"):
    for i in range(epochs):

        print("Epoch: 0")

        epoch_loss = 0
        epoch_truth_count = 0
        for idx, batch in enumerate(train_loader):
            loss, truth_count = train_loop(net, batch, loss_fn, optim, device)

            epoch_loss += loss
            epoch_truth_count += truth_count

            if idx % 1000 == 0:
                print(f"Loss: {loss} ({idx} / {len(train_loader)} x {i})")

        print(f"Epoch Loss: {epoch_loss}")
        print(f"Epoch Accuracy: {epoch_truth_count / len(train_loader)}")
    torch.save(net.state_dict(), "checkpoints/pytorch/version_1.pt")


def train_net_lightning(net, train_loader, validate_loader=None, epochs=10, checkpoint=None):
    if checkpoint is None:
        pl_net = LitTrainer(net)
    else:
        pl_net = load_pl_net(path=checkpoint)
    trainer = pl.Trainer(limit_train_batches=50, max_epochs=epochs,
                         default_root_dir="checkpoints/", accelerator="gpu")
    trainer.fit(pl_net, train_loader, validate_loader)


def load_pl_net(path="checkpoints/lightning_logs"):
    latest_version = os.listdir(path)[-1]
    checkpoint = f"{path}/{latest_version}/checkpoints/"
    checkpoint = checkpoint + os.listdir(checkpoint)[-1]

    print("Loading model weights from:", checkpoint)
    pl_net = LitTrainer.load_from_checkpoint(checkpoint)
    return pl_net


def load_torch_net(model, path="checkpoints/pytorch/version_0.pt"):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
