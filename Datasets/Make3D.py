import os.path
from .util import split2list
from .listdataset_test import ListDataset
from random import shuffle
import glob


def Make3D(split, **kwargs):
    input_root = kwargs.pop('root')
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    co_transform = kwargs.pop('co_transform', None)
    data_name = 'Make3D'

    print(glob.glob('images/*.png'))

    images = glob.glob(os.path.join(input_root, '*.jpg'))
    images = [[[os.path.basename(line), os.path.basename(line)], ['depth_sph_corr' + os.path.basename(line)[3:-3]+'mat']] for line in images]

    [train_list, test_list] = split2list(images, split)
    train_dataset = ListDataset(input_root, input_root, train_list, data_name=data_name, disp=True, of=False, transform=transform,
                                target_transform=target_transform, co_transform=co_transform)
    shuffle(test_list)
    test_dataset = ListDataset(input_root, input_root, test_list, disp=False, of=False, transform=transform,
                               target_transform=target_transform)
    return train_dataset, test_dataset
