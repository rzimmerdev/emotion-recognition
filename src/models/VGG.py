from torch import nn


class CNNFlow(nn.Module):  # VGG
    def __init__(self, input_channels, num_classes):
        super().__init__()

        self.feature_layers = [(input_channels, 64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512)]

        self.convolutions = []
        for block in self.feature_layers:

            block_channels = list(zip(block[:-1], block[1:]))

            layer = [nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())
                     for in_channels, out_channels in block_channels]

            self.convolutions.append(nn.Sequential(*layer))
        self.convolutions = nn.Sequential(*self.convolutions)

        self.linear = nn.Linear(in_features=3, out_features=num_classes)

    def forward(self, x):
        x = self.convolutions(x)
        x = x.reshape(x.size(0), -1)
        y_hat = self.linear(x)

        return y_hat
