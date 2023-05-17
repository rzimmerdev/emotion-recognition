import os

import pytorch_lightning as pl

from lightning.trainer import LitTrainer


def load_model(path="../checkpoints/lightning_logs"):
    latest_version = os.listdir(path)[-1]
    checkpoint = f"{path}/{latest_version}/checkpoints/"
    checkpoint = checkpoint + os.listdir(checkpoint)[-1]

    print("Loading model weights from:", checkpoint)
    pl_net = LitTrainer.load_from_checkpoint(checkpoint)
    return pl_net


def train_model(model, train_loader, validate_loader=None, epochs=10, checkpoint="../checkpoints/",
                pretrained=False):
    if not pretrained:
        pl_net = LitTrainer(model)
    else:
        pl_net = load_model(checkpoint + "lightning_logs/")
    trainer = pl.Trainer(limit_train_batches=50, max_epochs=epochs,
                         default_root_dir=checkpoint, accelerator="gpu")
    trainer.fit(pl_net, train_loader, validate_loader)