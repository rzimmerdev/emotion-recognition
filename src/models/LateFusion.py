from torch import nn, concat

from .CNN import UNet, VGG, ResNet, InceptionNet
from .OpticalFlow import dense_optical_flow


class LateMultidimensionalFusion(nn.Module):  # Transformer(UNET + VGG(Gunnar-Farneback))
    def __init__(self, in_features=1, out_features=8):
        super().__init__()

        self.cnn_raw = ResNet(in_features=in_features)
        self.cnn_flow = InceptionNet(in_features=in_features)

        self.optical_flow = dense_optical_flow

        self.rnn = nn.LSTM(input_size=64, hidden_size=100, num_layers=2)
        self.out = nn.Linear(in_features=100, out_features=out_features)

    def forward(self, x):
        features = concat((self.cnn_raw(x), self.cnn_flow(self.optical_flow(x))))
        series = self.rnn(features)

        return self.out(series)
