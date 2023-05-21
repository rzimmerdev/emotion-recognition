import pytorch_lightning as pl

import torch
import torchmetrics


class LitModel(pl.LightningModule):
    def __init__(self, model, num_classes=8):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(device=self.device, dtype=torch.float32)
        y = y.to(device=self.device, dtype=torch.float32)

        intervals = -(-len(x[0]) // 10)

        y_pred = self.model(x)

        y = torch.argmax(torch.repeat_interleave(y, intervals, dim=0), dim=1)

        train_loss = self.loss(y_pred, y)

        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(device=self.device, dtype=torch.float32)
        y = y.to(device=self.device, dtype=torch.float32)

        intervals = -(-len(x[0]) // 10)

        y_pred = self.model(x)
        y = torch.repeat_interleave(y, intervals, dim=0)
        validate_loss = self.loss(y_pred, torch.argmax(y, dim=1))
        accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.num_classes).to(device=self.device)
        softmax = torch.nn.Softmax(dim=0)

        self.log('accuracy', accuracy(softmax(y_pred[-1]), y[-1]))
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
