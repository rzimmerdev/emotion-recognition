#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots

from src.loaders.dataset import DatasetFER
from src.training.train import get_dataloaders, load_pl_net


def get_sequence(model, min_prob=True):
    fig = make_subplots(rows=2, cols=5)
    i = 0

    while i < 9:
        x, y = next(data)
        predicted, p = predict(x, model)
        if not min_prob or (predicted == i and p > 0.95):
            img = np.flip(np.array(x.reshape(48, 48)), 0)
            fig.add_trace(px.imshow(img).data[0], row=int(i/5)+1, col=i % 5+1)
            i += 1
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
    dataset = DatasetFER()
    _, validate_loader, test_loader = get_dataloaders(dataset)

    data = iter(validate_loader)

    print("PyTorch Lightning Network")
    get_sequence(load_pl_net().to("cuda")).show()
    # print("Manual Network")
    # get_sequence(load_torch_net().to("cuda")).show()
