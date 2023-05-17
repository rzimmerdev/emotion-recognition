import torch
from torch import nn
import numpy as np


from src.loaders.datasets import DatasetChildEFES
from src.loaders.dataloaders import get_dataloaders
from src.lightning.utils import load_weights
from src.lightning.model import LitModel


from src.models.LateFusion import LateMultidimensionalFusion


def predict(x, model):
    y_pred = model(x).to(device="cpu").detach()
    predicted = np.array((torch.argmax(y_pred, dim=1)))
    p = np.array(torch.max(nn.functional.softmax(y_pred, dim=1), dim=1).values)
    return predicted, p


def test_sample(device):
    frames = torch.tensor(np.random.rand(1, 52, 3, 1920, 1080), device=device, dtype=torch.float32)
    model = LateMultidimensionalFusion(3, 8, device)

    print("Predicting sample frames from random set")
    preds, probs = predict(frames, model)
    print(preds, probs)


def test_dataset(device):
    dataset = DatasetChildEFES()
    _, validate_loader, test_loader = get_dataloaders(dataset)

    data = next(iter(validate_loader))
    features = torch.tensor(data, dtype=torch.float32, device=device)

    model = LateMultidimensionalFusion(3, 8, device)
    pl_model = LitModel(model, num_classes=8)
    load_weights(pl_model, path="../checkpoints/lightning_logs")

    print("Predicting sample frames from validation set (ChildEFES dataset)")
    emotions, _ = predict(features, pl_model)
    print(emotions)


if __name__ == "__main__":
    device = torch.device("cuda")

    try:
        test_sample("cuda" if torch.cuda.is_available() else "cpu")
    except RuntimeError:
        test_sample("cpu")


