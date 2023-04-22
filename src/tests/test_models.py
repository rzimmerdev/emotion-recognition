import unittest
from unittest import TestCase

import torch.cuda

from models.VGG import CNNFlow

from loaders.dataloaders import ImageDataloader
from loaders.datasets import DatasetFER


class VGG(TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = DatasetFER()
        cls.dataloader = ImageDataloader(dataset)

        label_shape = dataset.data_shape[1][0]
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        cls.device = device
        cls.model = CNNFlow(input_channels=1, num_classes=label_shape).to(device)

    def test_input(self):
        features, labels = next(iter(self.dataloader))
        features = features.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device, dtype=torch.float32)

        predicted = self.model(features)
        self.assertEqual(predicted.shape, labels.shape)
        self.assertEqual(predicted.dtype, labels.dtype)
