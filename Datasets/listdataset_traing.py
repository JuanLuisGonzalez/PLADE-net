import torch.utils.data as data
import torch

import os
import os.path
from imageio import imread
import numpy as np
import random

LR_DATASETS = []
L_DATASETS = []


# indexes for l models
L_index = 0
R_index = 1


def img_loader(input_root, path_img):
    return imread(os.path.join(input_root, path_img))


class ListDataset(data.Dataset):
    def __init__(self, input_root, target_root, path_list, disp=False, of=False, transform=None,
                 target_transform=None, co_transform=None, max_pix=100, reference_transform=None,
                 fix=False, read_matted=False):
        self.input_root = input_root
        self.target_root = target_root
        self.path_list = path_list
        self.transform = transform
        self.reference_transform = reference_transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.disp = disp
        self.of = of
        self.input_loader = img_loader
        self.max = max_pix
        self.fix_order = fix
        self.read_matted = read_matted

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        inputs, targets = self.path_list[index]
        file_lname = inputs[L_index]
        file_rname = inputs[R_index]
        file_base_name = os.path.basename(inputs[L_index])[:-4]

        if random.random() < 0.5 or self.fix_order:
            x_pix = self.max
            inputs = [self.input_loader(self.input_root, inputs[L_index]),
                      self.input_loader(self.input_root, inputs[R_index])]
            if self.read_matted:
                inputs.append(self.input_loader(self.input_root, file_lname[:-4] + '_dm.png'))
                inputs.append(self.input_loader(self.input_root, file_rname[:-4] + '_dm.png'))
        else:
            x_pix = -self.max
            inputs = [self.input_loader(self.input_root, inputs[R_index]),
                      self.input_loader(self.input_root, inputs[L_index])]
            if self.read_matted:
                inputs.append(self.input_loader(self.input_root, file_rname[:-4] + '_dm.png'))
                inputs.append(self.input_loader(self.input_root, file_lname[:-4] + '_dm.png'))

        # if self.reference_transform is not None:
        #     inputs[0] = self.reference_transform(inputs[0])
        if self.co_transform is not None:
            inputs, _ = self.co_transform(inputs, targets)
        if self.transform is not None:
            for i in range(2):
                inputs[i] = self.transform(inputs[i])

        # Extract grid
        grid = np.transpose(inputs[len(inputs) - 1], (2, 0, 1))
        grid = torch.from_numpy(grid.copy()).float()

        # Append matted disps
        if self.read_matted:
            dm1 = torch.from_numpy(inputs[2].copy()).float() / 255
            dm2 = torch.from_numpy(inputs[3].copy()).float() / 255
            dm1 = dm1.unsqueeze(0)
            dm2 = dm2.unsqueeze(0)
            return inputs[0:2], [grid], x_pix, [dm1, dm2], file_base_name, file_lname, file_rname

        return inputs[0:2], [grid], x_pix, file_base_name, file_lname, file_rname
