import torch

from models import CNNColor
from train import get_dataloaders, train_net_lightning, load_pl_net
from predict import predict, get_sequence
from dataset import DatasetFER


def train(device):
    dataset = DatasetFER()
    train_loader, validate_loader, test_loader = get_dataloaders(dataset)

    net = CNNColor(input_channels=1, num_classes=9).to(device)  # Grayscale
    epochs = 20

    train_net_lightning(net, train_loader, validate_loader, epochs)
                        # checkpoint="checkpoints/lightning_logs/version_0")


def test(device):
    dataset = DatasetFER()
    _, validate_loader, test_loader = get_dataloaders(dataset)

    net = load_pl_net(path="checkpoints/lightning_logs/version_2")

    x, y = next(iter(validate_loader))

    get_sequence(net, device)


if __name__ == "__main__":
    # train("cuda")
    test("cuda")

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