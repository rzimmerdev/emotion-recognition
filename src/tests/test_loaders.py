import unittest
from unittest import TestCase
from loaders.datasets import DatasetFER, DatasetChildEFES
from loaders.dataloaders import ImageDataloader


class TestFER(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = DatasetFER()

    def test_shape(self):
        x, y = self.dataset.__getitem__(0)
        self.assertEqual(x.shape, (1, 48, 48))
        self.assertEqual(y.shape, (8,))  # 7 basic emotions + 1 neutral

    def test_size(self):
        self.assertEqual(len(self.dataset), 35887)  # 28709 + 3589 + 3589

    def test_loader(self):
        loader = ImageDataloader(self.dataset)
        features, labels = next(iter(loader))


class TestChildEFES(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = DatasetChildEFES(path="../../data/datasets/ChildEFES")

    def test_shape(self):
        x, y = self.dataset.__getitem__(0)
        self.assertEqual((3, 720, 1280), x.shape[1:])
        self.assertEqual(y.shape, (8,))

    def test_size(self):
        self.assertEqual(len(self.dataset), 168)


if __name__ == '__main__':
    unittest.main()
