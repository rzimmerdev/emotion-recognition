# from src.models.__init__ import CNNColor
# from src.training.train import get_dataloaders, train_net_lightning, load_pl_net
# from predict import get_sequence
# from src.loaders.dataset import DatasetFER
#
#
# def train(device):
#     dataset = DatasetFER()
#     train_loader, validate_loader, test_loader = get_dataloaders(dataset)
#
#     net = CNNColor(input_channels=1, num_classes=9).to(device)  # Grayscale
#     epochs = 20
#
#     train_net_lightning(net, train_loader, validate_loader, epochs)
#                         # checkpoint="checkpoints/lightning_logs/version_0")
#
#
# def test(device):
#     dataset = DatasetFER()
#     _, validate_loader, test_loader = get_dataloaders(dataset)
#
#     net = load_pl_net(path="checkpoints/lightning_logs/version_2")
#
#     x, y = next(iter(validate_loader))
#
#     get_sequence(net, device)
import torch


if __name__ == "__main__":
    print(torch.cuda.is_available())
