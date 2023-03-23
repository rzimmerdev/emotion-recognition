#!/usr/bin/env python
# coding: utf-8
import pytorch_lightning as pl

from torchmetrics import Accuracy
import torch.nn


class LitTrainer(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.model(x).reshape(1, -1)
        train_loss = self.loss(y_pred, y)

        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch

        y_pred = self.model(x).reshape(1, -1)
        validate_loss = self.loss(y_pred, y)

        accuracy = Accuracy(task="multiclass", num_classes=9)
        acc = accuracy(y_pred, y)
        self.log('accuracy', acc, on_epoch=True)

        self.log("val_loss", validate_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch

        y_pred = self.model(x).reshape(1, -1)
        test_loss = self.loss(y_pred, y)

        self.log("test_loss", test_loss)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)