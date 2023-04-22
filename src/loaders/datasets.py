#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class DatasetFER(Dataset):
    def __init__(self, path="../../data/FER"):
        self.path = path
        images = pd.read_csv(self.path + "/fer2013.csv")
        labels = pd.read_csv(self.path + "/fer2013new.csv")

        self.images = images["pixels"]
        self.labels = labels.iloc[:, 2:9]

        self.__data_shape = ((1, 48, 48), (7,))

    @property
    def data_shape(self):
        return self.__data_shape

    def __getitem__(self, n):
        data = self.images.iloc[n]
        label = self.labels.iloc[n]

        x = np.reshape(list(map(np.uint8, data.split(' '))), (1, 48, 48))
        y = np.array(label)

        return x, y

    def __len__(self):
        return len(self.labels)


class DatasetKDEF(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
