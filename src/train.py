import torch.cuda

from src.models.LateFusion import LateMultidimensionalFusion
from src.lightning.load import train_model
from src.loaders.dataloaders import VideoDataloader, get_dataloaders
from src.loaders.datasets import DatasetChildEFES


def train(device):
    dataset = DatasetChildEFES()
    train_loader, validate_loader, test_loader = get_dataloaders(dataset, VideoDataloader)

    model = LateMultidimensionalFusion(in_features=3, out_features=8).to(device)
    epochs = 20

    train_model(model, train_loader, validate_loader, epochs)


if __name__ == "__main__":
    train("cuda" if torch.cuda.is_available() else "cpu")
