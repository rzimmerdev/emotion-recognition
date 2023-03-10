#!/usr/bin/env python
# coding: utf-8
import pytorch_lightning as pl


class LitTrainer(pl.LightningModule):
    def __init__(self, model, optim, loss_fn):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.optim = optim
        self.loss = loss_fn

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
        return self.optim
