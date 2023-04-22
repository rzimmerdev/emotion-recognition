# # https://d2l.ai/chapter_recurrent-neural-networks/index.html
# import torch
# import torch.nn as nn
#
# import cv2 as cv
# import numpy as np
#
#
# class ImageRNN(nn.Module):
#     def __init__(self, input_channels, layers, output_channels, batch_size):
#         super(ImageRNN, self).__init__()
#
#         self.input_channels = input_channels
#         self.output_channels = output_channels
#         self.hidden = None
#         self.batch_size = batch_size
#
#         self.layers = layers
#
#         self.single_layer = nn.RNN(self.input_channels, layers[0])
#         self.linear = nn.Linear(self.layers[0], self.output_channels)
#
#     def empty(self):
#         return torch.zeros(1, self.batch_size, self.layers[0])
#
#     def forward(self, X):
#         X = X.permute(1, 0, 2)
#
#         self.hidden = self.empty()
#
#         out, self.hidden = self.single_layer(X, self.hidden)
#         out = self.linear(self.hidden)
#
#         return out.view(-1, self.output_channels)
