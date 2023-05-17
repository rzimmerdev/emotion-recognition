from unittest import TestCase

import numpy as np
import torch

from models.CNN import VGG, ResNet
from models.LateFusion import dense_optical_flow
from loaders.dataloaders import ImageDataloader, VideoDataloader
from loaders.datasets import DatasetFER, DatasetChildEFES


class StaticTestCase(TestCase):
    def setUp(self) -> None:
        self.dataset = DatasetFER()
        self.dataloader = ImageDataloader(self.dataset)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def get_single(self):
        if self.device is None:
            self.device = "cpu"
        features, labels = next(iter(self.dataloader))
        features = features.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device, dtype=torch.float32)
        return features, labels


class DynamicTestCase(TestCase):
    def setUp(self) -> None:
        self.dataset = DatasetChildEFES()
        self.dataloader = VideoDataloader(self.dataset)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def get_batch(self):
        if self.device is None:
            self.device = "cpu"
        batch, labels = next(iter(self.dataloader))
        batch = batch.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device, dtype=torch.float32)
        return batch, labels, batch.shape[1]


class VGGTestCase(StaticTestCase):

    def setUp(self):
        super().setUp()
        label_shape = self.dataset.data_shape[1][0]
        self.model = VGG(input_channels=1, num_classes=label_shape).to(self.device)

    def test_input(self):
        features, labels = self.get_single()

        predicted = self.model(features)
        self.assertEqual(predicted.shape, labels.shape)
        self.assertEqual(predicted.dtype, labels.dtype)


class ResNetTest(DynamicTestCase):

    def setUp(self) -> None:
        super().setUp()

        self.model = ResNet()

    def test_input(self):
        batch, labels, batch_size = self.get_batch()

        predicted = self.model(batch)
        self.assertEqual(predicted.shape, labels.shape)
        self.assertEqual(predicted.dtype, labels.dtype)


class OpticalFlowTest(DynamicTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.device = "cpu"
        self.optical_flow = dense_optical_flow

    def test_input(self):
        batch, label, batch_size = self.get_batch()
        batch = np.squeeze(batch, axis=0).transpose(1, 2, 0)
        label = np.squeeze(label, axis=0)
        flow = dense_optical_flow(batch)
        print(flow)
