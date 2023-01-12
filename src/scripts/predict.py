#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots

from src.scripts.trainer import LitTrainer
from src.scripts.models import CNN
from src.scripts.dataset import DatasetMNIST, download_mnist


def load_pl_net(path="checkpoints/lightning_logs/version_26/checkpoints/epoch=9-step=1000.ckpt"):
    pl_net = LitTrainer.load_from_checkpoint(path, model=CNN(1, 10))
    return pl_net


def load_torch_net(path="checkpoints/pytorch/version_0.pt"):
    state_dict = torch.load(path)
    net = CNN(1, 10)
    net.load_state_dict(state_dict)
    return net


def get_sequence(model):
    fig = make_subplots(rows=2, cols=5)

    i, j = 0, np.random.randint(0, 30000)

    while i < 10:
        x, y = dataset[j]

        predicted, p = predict(x, model)

        if predicted == i and p > 0.95:
            img = np.flip(np.array(x.reshape(28, 28)), 0)
            fig.add_trace(px.imshow(img).data[0], row=int(i/5)+1, col=i % 5+1)
            i += 1
        j += 1
    return fig


def predict(x, model, device="cuda"):
    y_pred = model(x.to(device)).detach().cpu()
    predicted = int(np.argmax(y_pred))
    p = torch.max(nn.functional.softmax(y_pred, dim=0))

    return predicted, p


def predict_interval(x, model, device="cuda"):
    y_pred = model(x.to(device))

    print(y_pred)

    predicted = np.argsort(y_pred.cpu().detach().numpy())
    p = nn.functional.softmax(y_pred, dim=0)

    return {int(i): float(p[i]) for i in predicted}


if __name__ == "__main__":
    mnist = download_mnist("downloads/mnist/")
    dataset, test_data = DatasetMNIST(*mnist["train"]), DatasetMNIST(*mnist["test"])

    print("PyTorch Lightning Network")
    get_sequence(load_pl_net().to("cuda")).show()
    print("Manual Network")
    get_sequence(load_torch_net().to("cuda")).show()
