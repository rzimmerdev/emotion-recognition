import torch
from torch import nn
import numpy as np


from src.loaders.datasets import DatasetChildEFES
from src.loaders.dataloaders import get_dataloaders
from src.lightning.load import load_model


def predict(x, model):
    y_pred = model(x).detach().cpu()
    predicted = int(np.argmax(y_pred, axis=1))
    p = torch.max(nn.functional.softmax(y_pred, dim=1), dim=1)
    return predicted, p

if __name__ == "__main__":
    dataset = DatasetChildEFES()
    _, validate_loader, test_loader = get_dataloaders(dataset)

    data = iter(validate_loader)

    print("Predicting sample frames from validation set (ChildEFES dataset)")
    predict(torch.tensor(data, dtype=torch.float32, device="cuda"), load_model().to("cuda"))
