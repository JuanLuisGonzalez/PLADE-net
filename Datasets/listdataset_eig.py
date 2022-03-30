import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np


def img_loader(input_root, path_imgs, index):
    imgs = [os.path.join(input_root, path) for path in path_imgs]
    return imread(imgs[index])


def depth_loader(input_root, path_depth):
    depth = np.load(os.path.join(input_root, path_depth))
    return depth[:, :, np.newaxis]


class ListDataset(data.Dataset):
    def __init__(self, input_root, target_root, path_list, transform=None, target_transform=None, co_transform=None):
        self.input_root = input_root
        self.target_root = target_root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

        self.input_loader = img_loader
        self.target_loader = depth_loader

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        inputs, targets = self.path_list[index]
        file_name = os.path.basename(inputs[0])[:-4]

        inputs = [self.input_loader(self.input_root, inputs, 0),
                  self.input_loader(self.input_root, inputs, 1)]

        targets = [self.target_loader(self.input_root, targets[0])]

        if self.co_transform is not None:
            inputs, targets = self.co_transform(inputs, targets)
        if self.transform is not None:
            for i in range(len(inputs)):
                inputs[i] = self.transform(inputs[i])
        if self.target_transform is not None:
            for i in range(len(targets)):
                targets[i] = self.target_transform(targets[i])

        return inputs, targets, file_name
