# sys
import os
import sys
import numpy as np
import random
import pickle

# image preprocess
import cv2

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms


class Feeder(torch.utils.data.Dataset):
    """ Feeder for AU recognition
    Arguments:
        label_path: the path to label
        image_path: the path to image
        debug: If true, only use the first 100 samples
    """
    def __init__(self,
                 label_path,
                 image_path,
                 debug=False,
                 image_size=256,
                 istrain=False):

        self.debug = debug
        self.label_path = label_path
        self.image_path = image_path
        self.image_size = image_size
        self.istrain = istrain

        self.load_data()

    def load_data(self):
        # data: N C H W

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load image
        with open(self.image_path, 'rb') as f:
            self.sample_name, self.imagepaths = pickle.load(f)

        if self.debug:
            self.label = self.label[0:100]
            self.imagepaths = self.imagepaths[0:100]
            self.sample_name = self.sample_name[0:100]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        label = np.array(self.label[index][-1])

        if 'BP4D' in self.imagepaths[index][0]:
            if 'ac512' in self.imagepaths[index][0]:
                subject_id = self.imagepaths[index][0].split('ac512/')[-1].split('/T')[0]
            else:
                subject_id = self.imagepaths[index][0].split('ac/')[-1].split('/T')[0]
        elif 'DISFA' in self.imagepaths[index][0]:
            if 'ac512' in self.imagepaths[index][0]:
                subject_id = self.imagepaths[index][0].split('ac512/')[-1][0:5]
            else:
                subject_id = self.imagepaths[index][0].split('ac/')[-1][0:5]

        image = []
        for i in range(len(self.imagepaths[index])):
            img = cv2.imread(self.imagepaths[index][i].replace(
                "/home/ubuntu/Documents", "../..",
                1).replace("../../..", "../..", 1))
            face = cv2.resize(img, (self.image_size, self.image_size))
            face = face.transpose((2, 0, 1))
            image.append(face / 255.0)

        image = np.array(image)

        return image, label, subject_id