import unittest
from unittest import TestCase
from loaders.datasets import DatasetFER, DatasetKDEF
from loaders.dataloaders import ImageDataloader


class TestFER(TestCase):
    def setUp(self):
        self.dataset = DatasetFER()

    def test_shape(self):
        x, y = self.dataset.__getitem__(0)
        self.assertEqual(x.shape, (48, 48))
        self.assertEqual(y.shape, (7,))  # 6 basic emotions + 1 neutral

    def test_size(self):
        self.assertEqual(len(self.dataset), 35887)  # 28709 + 3589 + 3589

    def test_loader(self):
        loader = ImageDataloader(self.dataset)



class TestKDEF(TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
