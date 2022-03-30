import os.path
from .util import split2list
from .listdataset_train import ListDataset
from .listdataset_traing import ListDataset as gridListDataset
from random import shuffle


def Kitti(split, **kwargs):
    input_root = kwargs.pop('root')
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    reference_transform = kwargs.pop('reference_transform', None)
    co_transform = kwargs.pop('co_transform', None)
    shuffle_test_data = kwargs.pop('shuffle_test_data', False)
    max_pix = kwargs.pop('max_pix', 100)
    fix = kwargs.pop('fix', False)
    use_grid = kwargs.pop('use_grid', True)
    train_split = kwargs.pop('train_split', 'eigen_train_split')
    read_matted_depth = kwargs.pop('read_matted_depth', False)

    # From Eigen et. al (NeurIPS 2014)
    if train_split == 'eigen_train_split':
        with open("Datasets/kitti_eigen_train.txt", 'r') as f:
            train_list = list(f.read().splitlines())
            if read_matted_depth:
                train_list = [[line.split(" "), None] for line in train_list if
                              os.path.isfile(os.path.join(input_root, line.split(" ")[0])) and
                              os.path.isfile(os.path.join(input_root, line.split(" ")[0][:-4] + '_dm.png'))]
            else:
                train_list = [[line.split(" "), None] for line in train_list if
                              os.path.isfile(os.path.join(input_root, line.split(" ")[0]))]
    # From Godard et. al (CVPR 2017)
    elif train_split == 'kitti_train_split':
        with open("Datasets/kitti_train_files.txt", 'r') as f:
            train_list = list(f.read().splitlines())
            if read_matted_depth:
                train_list = [[line.split(" "), None] for line in train_list if
                              os.path.isfile(os.path.join(input_root, line.split(" ")[0])) and
                              os.path.isfile(os.path.join(input_root, line.split(" ")[0][:-4] + '_dm.png'))]
            else:
                train_list = [[line.split(" "), None] for line in train_list if
                              os.path.isfile(os.path.join(input_root, line.split(" ")[0]))]

    train_list, test_list = split2list(train_list, split)

    if use_grid:
        train_dataset = gridListDataset(input_root, input_root, train_list, disp=False, of=False,
                                    transform=transform, target_transform=target_transform,
                                    co_transform=co_transform, reference_transform=reference_transform,
                                    max_pix=max_pix, fix=fix, read_matted=read_matted_depth)
        if shuffle_test_data:
            shuffle(test_list)
        test_dataset = gridListDataset(input_root, input_root, test_list, disp=False, of=False,
                                   transform=transform, target_transform=target_transform, fix=fix)
    else:
        train_dataset = ListDataset(input_root, input_root, train_list, disp=False, of=False,
                                    transform=transform, target_transform=target_transform,
                                    co_transform=co_transform,
                                    max_pix=max_pix, reference_transform=reference_transform, fix=fix)
        if shuffle_test_data:
            shuffle(test_list)
        test_dataset = ListDataset(input_root, input_root, test_list, disp=False, of=False,
                                   transform=transform, target_transform=target_transform, fix=fix)
    return train_dataset, test_dataset


def Kitti_list(split, **kwargs):
    input_root = kwargs.pop('root')
    with open('kitti_train_files.txt', 'r') as f:
        train_list = list(f.read().splitlines())
    train_list, test_list = split2list(train_list, split)
    return train_list, test_list
