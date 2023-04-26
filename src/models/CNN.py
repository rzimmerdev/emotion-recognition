import torch
from torch import nn, flatten


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


class ResNet(nn.Module):
    def __init__(self, in_features=1):
        super().__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)

        self.ap = nn.AdaptiveAvgPool2d(in_features)
        self.mp = nn.AdaptiveMaxPool2d(in_features)

    def forward(self, x):
        return self.model(torch.cat([self.mp(x), self.ap(x)], 1))


class InceptionNet(nn.Module):
    def __init__(self, in_features=1):
        super().__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False)

        self.ap = nn.AdaptiveAvgPool2d(in_features)
        self.mp = nn.AdaptiveMaxPool2d(in_features)

    def forward(self, x):
        return self.model(torch.cat([self.mp(x), self.ap(x)], 1))
