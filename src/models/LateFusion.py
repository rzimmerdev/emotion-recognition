from torch import nn, concat

from UNet import CNNColor
from VGG import CNNFlow
from OpticalFlow import dense_optical_flow
from Sequential import TransformerClassifier


class LateMultidimensionalFusion(nn.Module):  # Transformer(UNET + VGG(Gunnar-Farneback))
    def __init__(self, channels=1, classes=9):
        super().__init__()

        self.cnn_1 = CNNColor(channels, classes)
        self.cnn_2 = CNNFlow(channels, classes)

        self.optical_flow = dense_optical_flow

        self.transformer = TransformerClassifier(classes)

    def forward(self, x):
        features = concat((self.cnn_1(x), self.cnn_2(self.optical_flow(x))))

        return self.transformer(features)
