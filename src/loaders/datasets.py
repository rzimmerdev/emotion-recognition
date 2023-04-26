#!/usr/bin/env python
# coding: utf-8
import glob

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class DatasetFER(Dataset):
    def __init__(self, path="../data/datasets/FER"):
        self.path = path
        images = pd.read_csv(self.path + "/fer2013.csv")
        labels = pd.read_csv(self.path + "/fer2013new.csv")

        self.images = images["pixels"]
        self.labels = labels.iloc[:, 2:10]

        self.__data_shape = ((1, 48, 48), (8,))

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


class DatasetChildEFES(Dataset):
    def __init__(self, path="../data/datasets/ChildEFES"):
        self.path = path

        self.videos = glob.glob(self.path + "/*.mp4")
        self.images = glob.glob(self.path + "/*.jpg")

    def __getitem__(self, item):
        raw_path = self.videos[item]

        video = cv2.VideoCapture(raw_path)
        loading = True

        x_sequence = []
        y_raw = raw_path.split("/")[-1].split("_")[-1].split(".")[0]

        while loading:
            loading, frame = video.read()
            if not loading:
                break
            frame = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), axis=0)
            x_sequence.append(frame)

        x = np.asarray(x_sequence)
        y = np.zeros((8,))
        y[
            {"neutral": 0,
             "happy": 1,
             "surprise": 2,
             "sad": 3,
             "anger": 4,
             "disgust": 5,
             "fear": 6,
             "contempt": 7,
             }[y_raw]
        ] = 1

        return x, y

    def __len__(self):
        return len(self.videos)
