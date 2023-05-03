import torch.cuda

from src.models.LateFusion import LateMultidimensionalFusion
from src.training.train import train_net_lightning, load_pl_net
from src.loaders.dataloaders import VideoDataloader, get_dataloaders
from src.loaders.datasets import DatasetFER, DatasetChildEFES
from src.predict import predict


def train(device):
    dataset = DatasetChildEFES()
    train_loader, validate_loader, test_loader = get_dataloaders(dataset, VideoDataloader)

    net = LateMultidimensionalFusion(in_features=1, out_features=8).to(device)
    epochs = 20

    train_net_lightning(net, train_loader, validate_loader, epochs,
                        checkpoint="checkpoints/lightning_logs/version_0")


if __name__ == "__main__":
    train("cuda" if torch.cuda.is_available() else "cpu")
