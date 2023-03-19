# https://d2l.ai/chapter_recurrent-neural-networks/index.html
import torch
from torch import nn

import cv2 as cv
import numpy as np


class CNN_RGB(nn.Module):  # UNET
    def __init__(self, input_channels, num_classes):
        super().__init__()

        self.feature_layers = [input_channels, 32, 64]
        self.kernels = [5, 5]
        self.pools = [2, 3]

        self.feature_activations = [nn.ReLU for _ in range(len(self.feature_layers) - 1)]

        self.classifier_layers = [(48 // (2 * 3)) ** 2 * self.feature_layers[-1],
                                  120, 84, num_classes]
        self.classifier_activations = [nn.ReLU for _ in range(len(self.classifier_layers) - 1)]

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
        y = self.classifier(torch.flatten(x))
        return y


class CNN_HSV(nn.Module):  # VGG
    def __init__(self):
        super().__init__()

        self.convolutions = nn.Module()
        self.linear = nn.Module()

    def forward(self, x):
        x = self.convolutions(x)
        x = x.reshape(x.size(0), -1)
        y_hat = self.linear(x)

        return y_hat

class DenseOpticalFlow:  # Gunnar-Farneback
    def __init__(self):
        pass

    def forward(self, x):
        if len(x) <= 0:
            return x

        mask = np.zeros_like(x)

        curr_mask = np.zeros_like(x[0])
        curr_mask[..., 1] = 255

        prev = np.zeros_like(x[0])
        prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

        for idx, frame in enumerate(x):
            curr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            flow = cv.calcOpticalFlowFarneback(curr, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
            curr_mask[..., 0] = angle * 90 / np.pi
            curr_mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

            mask[idx] = curr_mask

        return mask

class TransformerClassifier(nn.Module):  # Simple Default Transformer Layer
    pass


class LateMultidimensionalFusion(nn.Module):  # Transformer(UNET + VGG(Gunnar-Farneback))
    pass


class ImageRNN(nn.Module):
    def __init__(self, input_channels, layers, output_channels, batch_size):
        super(ImageRNN, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden = None
        self.batch_size = batch_size

        self.layers = layers

        self.single_layer = nn.RNN(self.input_channels, layers[0])
        self.linear = nn.Linear(self.layers[0], self.output_channels)

    def empty(self):
        return torch.zeros(1, self.batch_size, self.layers[0])

    def forward(self, X):
        X = X.permute(1, 0, 2)

        self.hidden = self.empty()

        out, self.hidden = self.single_layer(X, self.hidden)
        out = self.linear(self.hidden)

        return out.view(-1, self.output_channels)
