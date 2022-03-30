# Copyright (C) 2021  Juan Luis Gonzalez Bello (juanluisgb@kaist.ac.kr)
# This software is not for commercial use
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from __future__ import division
import torch
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage
from PIL import Image
import torch.nn.functional as F

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input, target = t(input, target)
        return input, target


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert (isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array.copy())
        # put it from HWC to CHW format
        return tensor.float()


class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, targets=None):
        h1, w1, _ = inputs[0].shape
        h2, w2, _ = targets[0].shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        x2 = int(round((w2 - tw) / 2.))
        y2 = int(round((h2 - th) / 2.))

        for i in range(len(inputs)):
            inputs[i] = inputs[i][y1: y1 + th, x1: x1 + tw]
        if targets is not None:
            for i in range(len(targets)):
                targets[i] = targets[i][y2: y2 + th, x2: x2 + tw]
        return inputs, targets


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, targets=None):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs, targets

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for i in range(len(inputs)):
            inputs[i] = inputs[i][y1: y1 + th, x1: x1 + tw]
        if targets is not None:
            for i in range(len(targets)):
                targets[i] = targets[i][y1: y1 + th, x1: x1 + tw]
        return inputs, targets


class RandomResizeCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, down, up):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.s_factor = (down, up)

    def __call__(self, inputs, targets=None):
        h, w, _ = inputs[0].shape
        th, tw = self.size

        min_factor = max(max((th + 1) / h, (tw + 1) / w), self.s_factor[0])  # plus one to ensure
        max_factor = self.s_factor[1]
        factor = np.random.uniform(low=min_factor, high=max_factor)

        for i in range(len(inputs)):
            input = Image.fromarray(inputs[i]).resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
            inputs[i] = np.array(input)
        if targets is not None:
            for i in range(len(targets)):
                target = Image.fromarray(targets[i]).resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
                targets[i] = np.array(target)

        # get grid
        i_tetha = torch.zeros(1, 2, 3)
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        a_grid = F.affine_grid(i_tetha, torch.Size([1, 3, int(h * factor), int(w * factor)]), align_corners=True)
        inputs.append(a_grid[0, :, :, :].numpy())

        h, w, _ = inputs[0].shape
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for i in range(len(inputs)):
            inputs[i] = inputs[i][y1: y1 + th, x1: x1 + tw]
        if targets is not None:
            for i in range(len(targets)):
                targets[i] = targets[i][y1: y1 + th, x1: x1 + tw]
        return inputs, targets


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
        if doing this on disparity estimation you need both disparities left and right need stereo targets
    """

    def __init__(self, disp=False, of=False):
        self.of = of
        self.disp = disp

    def __call__(self, inputs, targets=None):
        o_inputs = []
        o_target = []

        if random.random() < 0.5:
            if self.disp and self.of:
                o_inputs.append(np.copy(np.fliplr(inputs[1])))  # at t
                o_inputs.append(np.copy(np.fliplr(inputs[0])))
                o_inputs.append(np.copy(np.fliplr(inputs[3])))  # at t + 1
                o_inputs.append(np.copy(np.fliplr(inputs[2])))
                o_target.append(np.copy(np.fliplr(targets[1])))  # disp
                o_target.append(np.copy(np.fliplr(targets[0])))  # disp
                o_target.append(np.copy(np.fliplr(targets[2])))  # of
                o_target.append(np.copy(np.fliplr(targets[3])))  # of
                o_target[2][:, :, 0] *= -1
                o_target[3][:, :, 0] *= -1
                return o_inputs, o_target
            if self.disp:
                o_inputs.append(np.copy(np.fliplr(inputs[1])))
                o_inputs.append(np.copy(np.fliplr(inputs[0])))
                o_target.append(np.copy(np.fliplr(targets[1])))
                o_target.append(np.copy(np.fliplr(targets[0])))
                return o_inputs, o_target
            if self.of:
                for i in range(len(inputs)):
                    inputs[i] = np.copy(np.fliplr(inputs[i]))
                for i in range(len(targets)):
                    targets[i] = np.copy(np.fliplr(targets[i]))
                    targets[i][:, :, 0] *= -1
                return inputs, targets
            else:  # only lr
                o_inputs.append(np.copy(np.fliplr(inputs[1])))
                o_inputs.append(np.copy(np.fliplr(inputs[0])))
                return o_inputs, targets
        else:
            return inputs, targets


class RandomHorizontalFlipG(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
        if doing this on disparity estimation you need both disparities left and right need stereo targets
    """

    def __init__(self, disp=False, of=False):
        self.of = of
        self.disp = disp

    def __call__(self, inputs, targets=None):
        o_inputs = []
        o_target = []

        if random.random() < 0.5:
            o_inputs.append(np.copy(np.fliplr(inputs[1])))
            o_inputs.append(np.copy(np.fliplr(inputs[0])))

            # Invert depth mates, if used
            if len(inputs) > 3:
                o_inputs.append(np.copy(np.fliplr(inputs[3])))
                o_inputs.append(np.copy(np.fliplr(inputs[2])))

            # Invert grid sine in X axis
            inputs[len(inputs) - 1][:, :, 0] = -inputs[len(inputs) - 1][:, :, 0]
            o_inputs.append(np.copy(np.fliplr(inputs[len(inputs) - 1])))
            return o_inputs, targets
        else:
            return inputs, targets


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
        only optical flow is inverted
    """

    def __init__(self, stereo_targets, disp=False, of=False):
        self.of = of
        self.disp = disp
        self.stereo_targets = stereo_targets

    def __call__(self, inputs, targets=None):
        if random.random() < 0.5:
            for i in range(len(inputs)):
                inputs[i] = np.copy(np.flipud(inputs[i]))
            if self.disp or self.of:
                for i in range(len(targets)):
                    targets[i] = np.copy(np.flipud(targets[i]))
            if self.disp and self.of:
                if self.stereo_targets:
                    targets[2][:, :, 1] *= -1
                    targets[3][:, :, 1] *= -1
                else:
                    targets[1][:, :, 1] *= -1
            elif self.of:
                if self.stereo_targets:
                    targets[0][:, :, 1] *= -1
                    targets[1][:, :, 1] *= -1
                else:
                    targets[0][:, :, 1] *= -1
        return inputs, targets


class RandomTranslate(object):
    # use only on monocular optical flow estimation (only forward flow supported now)

    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return inputs, target
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1, x2, x3, x4 = max(0, tw), min(w + tw, w), max(0, -tw), min(w - tw, w)
        y1, y2, y3, y4 = max(0, th), min(h + th, h), max(0, -th), min(h - th, h)

        inputs[0] = inputs[0][y1:y2, x1:x2]
        inputs[1] = inputs[1][y3:y4, x3:x4]
        target[0] = target[0][y1:y2, x1:x2]
        target[0][:, :, 0] += tw
        target[0][:, :, 1] += th
        return inputs, target


class RandomDownUp(object):
    def __init__(self, max_down):
        self.down_factor = max_down

    def __call__(self, input):
        factor = np.random.uniform(low=1 / self.down_factor, high=1)
        h, w, _ = input.shape
        input = Image.fromarray(input)
        input = input.resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
        input = input.resize((int(w), int(h)), resample=Image.BICUBIC)
        input = np.array(input)
        return input


class Resize(object):
    def __init__(self, target_h, target_w):
        self.target_dim = (target_h, target_w)

    def __call__(self, inputs, targets=None):
        for i in range(len(inputs)):
            im_input = Image.fromarray(inputs[i]).resize((int(self.target_dim[1]), int(self.target_dim[0])),
                                                         resample=Image.BILINEAR)
            inputs[i] = np.array(im_input)
        if targets is not None:
            for i in range(len(targets)):
                im_target = Image.fromarray(targets[i]).resize((int(self.target_dim[1]), int(self.target_dim[0])),
                                                               resample=Image.BILINEAR)
                targets[i] = np.array(im_target)
        return inputs, targets


class CropEdge(object):
    """Crops edges of the given PIL.Image. size can be a tuple (crop_left, crop_right, crop_top, crop_bottom)
    or an integer, in which case the target will be of a square shape (size, size, size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, targets=None):
        h, w, _ = inputs[0].shape
        c_left, c_right, c_top, c_bot = self.size
        if c_left == 0 and c_right == 0 and c_top == 0 and c_bot == 0:
            return inputs, targets

        for i in range(len(inputs)):
            inputs[i] = inputs[i][c_top: h - c_bot, c_left: w - c_right]
        if targets is not None:
            for i in range(len(targets)):
                targets[i] = targets[i][c_top: h - c_bot, c_left: w - c_right]
        return inputs, targets


class RandomGamma(object):
    def __init__(self, min=1, max=1):
        self.min = min
        self.max = max
        self.A = 255

    def __call__(self, inputs, targets=None):
        if random.random() < 0.5:
            factor = random.uniform(self.min, self.max)
            for i in range(2):
                inputs[i] = self.A * ((inputs[i] / 255) ** factor)
            return inputs, targets
        else:
            return inputs, targets


class RandomBrightness(object):
    def __init__(self, min=0, max=0):
        self.min = min
        self.max = max

    def __call__(self, inputs, targets=None):
        if random.random() < 0.5:
            factor = random.uniform(self.min, self.max)
            for i in range(2):
                inputs[i] = inputs[i] * factor
                inputs[i][inputs[i] > 255] = 255
            return inputs, targets
        else:
            return inputs, targets


class RandomCBrightness(object):
    def __init__(self, min=0, max=0):
        self.min = min
        self.max = max

    def __call__(self, inputs, targets=None):
        if random.random() < 0.5:
            for i in range(2):
                for c in range(3):
                    factor = random.uniform(self.min, self.max)
                    inputs[i][:, :, c] = inputs[i][:, :, c] * factor
                inputs[i][inputs[i] > 255] = 255
            return inputs, targets
        else:
            return inputs, targets


class RandomCBrightness2(object):
    def __init__(self, min=0, max=0):
        self.min = min
        self.max = max

    def __call__(self, inputs, targets=None):
        if random.random() < 0.5:
            for c in range(3):
                factor = random.uniform(self.min, self.max)
                for i in range(2):
                    inputs[i][:, :, c] = inputs[i][:, :, c] * factor
                    inputs[i][inputs[i] > 255] = 255
            return inputs, targets
        else:
            return inputs, targets