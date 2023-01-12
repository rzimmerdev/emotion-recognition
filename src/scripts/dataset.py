#!/usr/bin/env python
# coding: utf-8
import gzip

import torch
from torch.utils.data import Dataset
import numpy as np

from src.downloader import download_dataset


def download_mnist(download_dir):
    download_dataset("mnist", download_dir)

    return {"train": (download_dir + "train_images", download_dir + "train_labels"),
            "test": (download_dir + "test_images", download_dir + "test_labels")}


class DatasetMNIST(Dataset):
    def __init__(self, images, labels):
        with gzip.open(images, 'r') as f:
            f.read(4)
            self.total = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            columns = int.from_bytes(f.read(4), 'big')

            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8).reshape((self.total, rows, columns))
            self.images = images
        with gzip.open(labels, 'r') as f:
            f.read(8)

            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)
            self.labels = labels
        self.data = list(zip(self.images, self.labels))

    def __getitem__(self, n):
        if n > self.total:
            raise ValueError(f"Dataset doesn't have enough elements to suffice request of {n} elements.")
        return torch.tensor(self.data[n][0].reshape(1, 28, 28), dtype=torch.float32), torch.tensor(self.data[n][1])

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    download_dir = "downloads/mnist/"
    mnist = download_mnist(download_dir)

    dataset = DatasetMNIST(*mnist["train"])

    import matplotlib.pyplot as plt

    X, y = dataset[4]
    plt.imshow(X, cmap="gray")
    plt.title(label="Annotated label: " + str(y))
    plt.show()
