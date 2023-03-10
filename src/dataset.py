#!/usr/bin/env python
# coding: utf-8
import gzip
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


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


class DatasetFER(Dataset):
    def __init__(self, path="/home/rzimmerdev/Downloads/fer2013"):
        self.path = path

        labels_path = path + "/fer2013new.csv"
        labels = pd.read_csv(labels_path)
        images = pd.read_csv(self.path + "/fer2013.csv")

        self.classes = list(labels.columns[2:])
        self.paths = labels["Image name"]
        self.labels = labels[self.classes]
        self.images = images["pixels"]

    def __getitem__(self, n):
        label = self.labels.iloc[n]
        data = self.images.iloc[n]
        image = np.reshape(list(map(np.float32, data.split(' '))),
                           (48, 48))
        return torch.tensor(image) / 10, torch.tensor(label, dtype=torch.float32)[:-1].reshape(-1)

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    # download_dir = "downloads/mnist/"
    # mnist = download_mnist(download_dir)
    #
    # dataset = DatasetMNIST(*mnist["train"])

    import matplotlib.pyplot as plt
    dataset = DatasetFER()
    X, y = dataset[0]
    plt.imshow(X, cmap="gray")
    plt.title(label="Annotated label: " + dataset.classes[np.argmax(y)])
    plt.show()
