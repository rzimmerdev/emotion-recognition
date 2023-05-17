import torch.cuda

from src.models.LateFusion import LateMultidimensionalFusion
from src.lightning.utils import train_model
from src.loaders.dataloaders import VideoDataloader, get_dataloaders
from src.loaders.datasets import DatasetChildEFES


def train(device):
    dataset = DatasetChildEFES()
    train_loader, validate_loader, test_loader = get_dataloaders(dataset, VideoDataloader)

    model = LateMultidimensionalFusion(3, 8, "cpu")
    epochs = 20

    train_model(model, train_loader, validate_loader, epochs, accelerator=device, pretrained=True)


if __name__ == "__main__":
    try:
        train("cuda" if torch.cuda.is_available() else "cpu")
    except RuntimeError:
        train("cpu")
