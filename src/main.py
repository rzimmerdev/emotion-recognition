import torch

from src.models import CNN_RGB
from src.train import get_dataloaders, train_net_lightning
from src.dataset import DatasetFER


def train(device):
    dataset = DatasetFER()
    train_loader, validate_loader, test_loader = get_dataloaders(dataset)

    net = CNN_RGB(input_channels=1, num_classes=9).to(device)
    epochs = 20

    train_net_lightning(net, train_loader, validate_loader, epochs,
                        checkpoint="checkpoints/lightning_logs/version_0")


if __name__ == "__main__":
    train("cuda")


# Optimizers:
#
# - Adam
# - SGD
#
#
# Loss:
#
# - RMSE
# - Cross-Entropy
#
#
# Layers:
#
# - Convolution
# - RNN / LSTM
# - Linear
