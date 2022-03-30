import os.path
from .util import split2list
from .listdataset_test import ListDataset
from random import shuffle


def Kitti_vdyne(split, **kwargs):
    input_root = kwargs.pop('root')
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    co_transform = kwargs.pop('co_transform', None)
    shuffle_test = kwargs.pop('shuffle_test', True)

    with open("Datasets/kitti_eigen_test_original.txt", 'r') as f:
        train_list = list(f.read().splitlines())
        train_list = [[line.split(" "), [line.split(" ")[0][0:-3] + 'npy', line.split(" ")[1][0:-3] + 'npy']]
                      for line in train_list if (
                              os.path.isfile(os.path.join(input_root, line.split(" ")[0][0:-3] + 'npy')) and
                              os.path.isfile(os.path.join(input_root, line.split(" ")[0])))]

    train_list, test_list = split2list(train_list, split)

    train_dataset = ListDataset(input_root, input_root, train_list,
                                transform=transform, target_transform=target_transform, co_transform=co_transform)
    if shuffle_test:
        shuffle(test_list)

    test_dataset = ListDataset(input_root, input_root, test_list,
                               transform=transform, target_transform=target_transform)

    return train_dataset, test_dataset
