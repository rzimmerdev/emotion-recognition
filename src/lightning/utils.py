import os

import pytorch_lightning as pl

from .model import LitModel


def load_weights(pl_model, path="../../checkpoints/lightning_logs"):
    print("Loading model weights from:", path)

    versions = os.listdir(path) if os.path.exists(path) else None

    if versions:
        latest_version = versions[-1]
        print(f"Using latest weights version: {latest_version}")
        checkpoint = f"{path}/{latest_version}/checkpoints/"
        checkpoint = checkpoint + os.listdir(checkpoint)[-1]

        pl_model.load_from_checkpoint(checkpoint)
    else:
        print("No checkpointed weights found, skipping...")


def train_model(model, train_loader, validate_loader=None, epochs=10, checkpoints="../../checkpoints/",
                pretrained=False, accelerator="cpu"):
    pl_model = LitModel(model)

    if pretrained:
        load_weights(pl_model, checkpoints + "lightning_logs/")

    trainer = pl.Trainer(limit_train_batches=50, max_epochs=epochs,
                         default_root_dir=checkpoints, accelerator=accelerator)
    trainer.fit(pl_model, train_loader, validate_loader)
