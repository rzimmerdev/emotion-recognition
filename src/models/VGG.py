from torch import nn


class CNNFlow(nn.Module):  # VGG
    def __init__(self, input_channels, num_classes):
        super().__init__()

        self.feature_layers = [[input_channels], [64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]

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
