import os.path
from random import shuffle
from .listdataset_eth3d import ListDataset
from .util import split2list


def make_dataset(main_dir):
    directories = []
    lr_folder = os.path.join(main_dir, 'MiddEval3-data-F', 'MiddEval3', 'trainingF')
    gt_folder = os.path.join(main_dir, 'MiddEval3-GT0-F', 'MiddEval3', 'trainingF')
    for lr_dirs in os.listdir(lr_folder):
        # Only read folders not files
        if os.path.isfile(os.path.join(lr_folder, lr_dirs)):
            continue

        imgl = os.path.join(lr_folder, lr_dirs, 'im0.png')      # rgb input left
        imgr = os.path.join(lr_folder, lr_dirs, 'im1.png')      # rgb input right
        imgd = os.path.join(gt_folder, lr_dirs, 'disp0GT.pfm')  # disp GT

        # Check valid files
        if not (os.path.isfile(imgl) and
                os.path.isfile(imgr) and
                os.path.isfile(imgd)):
            continue
        directories.append([[imgl, imgr], [imgd]])

    return directories


def Middlebury(split, **kwargs):
    input_root = kwargs.pop('root')
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    co_transform = kwargs.pop('co_transform', None)
    shuffle_test = kwargs.pop('shuffle_test', False)

    train_list = make_dataset(input_root)
    train_list, test_list = split2list(train_list, split)

    train_dataset = ListDataset(input_root, input_root, train_list, transform=transform,
                                   target_transform=target_transform, co_transform=co_transform)
    test_dataset = ListDataset(input_root, input_root, test_list, transform=transform,
                                  target_transform=target_transform)
    if shuffle_test:
        shuffle(test_list)

    return train_dataset, test_dataset
