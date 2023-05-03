import torch
from torch import nn, flatten
from torchvision import transforms
from torchvision.models import Inception3, resnet34


class UNet(nn.Module):  # UNET
    def __init__(self, input_channels, num_classes):
        super().__init__()

        self.feature_layers = [input_channels, 32, 64]
        self.kernels = [5, 5]
        self.pools = [2, 3]

        self.feature_activations = [nn.ReLU for _ in range(len(self.feature_layers) - 1)]

        self.classifier_layers = [(48 // (2 * 3)) ** 2 * self.feature_layers[-1],
                                  120, 84, num_classes]
        self.classifier_activations = [nn.ReLU for _ in range(len(self.classifier_layers) - 1)]
        self.out = nn.Softmax(dim=0)

        feature_layers = []
        for idx, layer in enumerate(list(zip(self.feature_layers[:-1], self.feature_layers[1:]))):
            feature_layers.append(
                nn.Conv2d(in_channels=layer[0], out_channels=layer[1],
                          kernel_size=self.kernels[idx], padding=self.kernels[idx] // 2)
            )
            feature_layers.append(self.feature_activations[idx]())
            feature_layers.append(nn.MaxPool2d(kernel_size=self.pools[idx]))

        classifier_layers = []
        for idx, layer in enumerate(list(zip(self.classifier_layers[:-1], self.classifier_layers[1:]))):
            classifier_layers.append(
                nn.Linear(in_features=layer[0], out_features=layer[1])
            )
            if idx < len(self.classifier_activations) - 1:
                classifier_layers.append(self.classifier_activations[idx]())

        self.feature_extractor = nn.Sequential(*feature_layers)
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        y = self.classifier(flatten(x))
        return self.out(y)


class VGG(nn.Module):  # VGG
    def __init__(self, input_channels, num_classes):
        super().__init__()

        self.feature_layers = [[input_channels], [64, 64], [128, 128],
                               [256, 256, 256], [512, 512, 512], [512, 512, 512]]

        self.convolutions = []
        for idx, block in enumerate(self.feature_layers[1:]):

            block_channels = list(zip(block[:-1], block[1:]))
            block_channels.insert(0, (self.feature_layers[idx][-1], block[0]))

            conv_relu_layers = [nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
                                              nn.BatchNorm2d(out_channels),
                                              nn.ReLU())
                                for in_channels, out_channels in block_channels]

            if idx < len(self.feature_layers) - 2:
                layer = nn.Sequential(*conv_relu_layers, nn.MaxPool2d((2, 2)))
            else:
                layer = nn.Sequential(*conv_relu_layers, nn.MaxPool2d((3, 3)))

            self.convolutions.append(nn.Sequential(*layer))
        self.convolutions = nn.Sequential(*self.convolutions)

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.feature_layers[-1][-1], out_features=256),
            nn.Linear(in_features=256, out_features=100),
            nn.Linear(in_features=100, out_features=num_classes)
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.reshape(-1, 512)
        y_hat = self.linear(x)

        return y_hat


class Channels(nn.Module):
    def __init__(self, channels, batched=True):
        super().__init__()
        if batched:
            self.forward = lambda x: x.repeat(1, channels, 1, 1)
        else:
            self.forward = lambda x: x.repeat(channels, 1, 1)


def preprocess(resize, crop, is_rgb=True):
    if not is_rgb:
        return transforms.Compose([
            Channels(3),
            transforms.Resize(resize, antialias=True),
            transforms.CenterCrop(crop),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(resize, antialias=True),
            transforms.CenterCrop(crop),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class ResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=8):
        super().__init__()

        model = resnet34()

        self.preprocess = preprocess(256, 224, is_rgb=in_channels == 3)

        self.model = nn.Sequential(
            model,
            nn.Linear(1000, out_channels)
        )

    def forward(self, x):
        return self.model(self.preprocess(x))


class InceptionNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=8):
        super().__init__()

        self.model = Inception3(num_classes=out_channels, init_weights=False)

    def forward(self, x):
        return self.model(x)
