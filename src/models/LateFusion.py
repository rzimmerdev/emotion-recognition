import numpy as np
from torch import nn, concat, tensor

from .CNN import ResNet, InceptionNet
from .OpticalFlow import dense_optical_flow


class LateMultidimensionalFusion(nn.Module):  # Transformer(UNET + VGG(Gunnar-Farneback))
    def __init__(self, in_features=1, out_features=8):
        super().__init__()

        self.cnn_raw = InceptionNet(in_features, 64)
        self.cnn_flow = ResNet(in_features, 64)

        self.optical_flow = lambda x: tensor(dense_optical_flow(x).astype(np.float32))

        self.rnn = nn.LSTM(input_size=128, hidden_size=256, num_layers=2)
        self.out = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.Linear(in_features=64, out_features=out_features)
        )

        self.softmax = nn.Softmax(dim=1)  # Apply softmax to each frame prediction

    def forward(self, x):
        features = concat((self.cnn_raw(x).logits, self.cnn_flow(self.optical_flow(x))), dim=1)
        series = self.rnn(features)[0]

        return self.softmax(self.out(series))
