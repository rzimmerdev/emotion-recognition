import torch

from src.models import CNN
from src.dataset import DatasetMNIST, download_mnist
from src.train import get_dataloaders, train_net_manually, train_net_lightning


def main(device):
    mnist = download_mnist("downloads/mnist/")
    dataset, test_data = DatasetMNIST(*mnist["train"]), DatasetMNIST(*mnist["test"])
    train_loader, validate_loader, test_loader = get_dataloaders(dataset, test_data)

    # Training manually
    net = CNN(input_channels=1, num_classes=10).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    max_epochs = 1

    train_net_manually(net, optim, loss_fn, train_loader, validate_loader, max_epochs, device)


if __name__ == "__main__":
    main("cpu")
