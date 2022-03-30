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

gpu = '0'

import argparse
import time
import numpy as np
from imageio import imsave
import matplotlib.pyplot as plt
from PIL import Image

import Datasets
import models

dataset_names = sorted(name for name in Datasets.__all__)
model_names = sorted(name for name in models.__all__)

parser = argparse.ArgumentParser(description='Testing pan generation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data', metavar='DIR', default='E:\\Datasets\\', help='path to dataset')
parser.add_argument('-tn', '--tdataName', metavar='Test Data Set Name', default='Kitti_eigen_test_improved',
                    choices=dataset_names)
parser.add_argument('-relbase', '--rel_baselne', default=1, help='Relative baseline of testing dataset')
parser.add_argument('-mdisp', '--max_disp', default=300)
parser.add_argument('-mindisp', '--min_disp', default=2)
parser.add_argument('-b', '--batch_size', metavar='Batch Size', default=1)
parser.add_argument('-eval', '--evaluate', default=True)
parser.add_argument('-save', '--save', default=True)
parser.add_argument('-save_pc', '--save_pc', default=False)
parser.add_argument('-save_pan', '--save_pan', default=False)
parser.add_argument('-save_input', '--save_input', default=True)
parser.add_argument('-save_gt', '--save_gt', default=True)
parser.add_argument('-w', '--workers', metavar='Workers', default=4)
parser.add_argument('--sparse', default=False, action='store_true',
                    help='Depth GT is sparse, automatically seleted when choosing a KITTIdataset')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')

# Kitti_stage2g\11-07-00_25\FAL_net2B_gep,e20es7800,b4,lr5e-05   # (K)
# Kitti_stage2g\10-31-10_23\FAL_net2B_gep,e11es7800,b4,lr5e-05   # (KCS)
# Kitti_stage2g\11-06-19_41\FAL_net2B_gep2s,e11es7800,b4,lr5e-05 # Stereo (K)
parser.add_argument('-dt', '--dataset', help='Dataset and training stage directory', default='Kitti_stage2g')
parser.add_argument('-ts', '--time_stamp', help='Model timestamp', default='10-31-10_23')
parser.add_argument('-m', '--model', help='Model', default='FAL_net2B_gep')
parser.add_argument('-no_levels', '--no_levels', default=49, help='Number of quantization levels in MED')
parser.add_argument('-use_grid', '--use_grid', default=True, help='use input identity grid')
parser.add_argument('-is_stereo', '--is_stereo', default=False, help='use input identity grid')
parser.add_argument('-dtl', '--details', help='details',
                    default=',e11es7800,b4,lr5e-05/checkpoint.pth.tar')
parser.add_argument('-fpp', '--f_post_process', default=False, help='Post-processing with flipped input')
parser.add_argument('-mspp', '--ms_post_process', default=False, help='Post-processing with multi-scale input')
parser.add_argument('-median', '--median', default=False,
                    help='use median scaling (not needed when training from stereo')


def display_config(save_path):
    settings = ''
    settings = settings + '############################################################\n'
    settings = settings + '# PLADE-net      -        Pytorch implementation           #\n'
    settings = settings + '# by Juan Luis Gonzalez   juanluisgb@kaist.ac.kr           #\n'
    settings = settings + '############################################################\n'
    settings = settings + '-------YOUR TRAINING SETTINGS---------\n'
    for arg in vars(args):
        settings = settings + "%15s: %s\n" % (str(arg), str(getattr(args, arg)))
    print(settings)
    # Save config in txt file
    with open(os.path.join(save_path, 'settings.txt'), 'w+') as f:
        f.write(settings)


def main():
    print('-------Testing on gpu ' + gpu + '-------')

    save_path = os.path.join('Test_Results', args.tdataName, args.model, args.time_stamp)
    if args.f_post_process:
        save_path = save_path + 'fpp'
    if args.ms_post_process:
        save_path = save_path + 'mspp'
    print('=> Saving to {}'.format(save_path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    display_config(save_path)

    input_transform = transforms.Compose([
        data_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),  # (input - mean) / std
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    target_transform = transforms.Compose([
        data_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0], std=[1]),
    ])

    # Torch Data Set List
    input_path = os.path.join(args.data, args.tdataName)
    [test_dataset, _] = Datasets.__dict__[args.tdataName](split=1,  # all to be tested
                                                          root=input_path,
                                                          disp=True,
                                                          shuffle_test=False,
                                                          transform=input_transform,
                                                          target_transform=target_transform)

    # Torch Data Loader
    args.batch_size = 1  # kitty mixes image sizes!
    args.sparse = True  # disparities are sparse (from lidar)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=False, shuffle=False)

    # create pan model
    model_dir = os.path.join(args.dataset, args.time_stamp, args.model + args.details)
    pan_network_data = torch.load(model_dir)
    pan_model = pan_network_data['m_model']
    print("=> using pre-trained model for pan '{}'".format(pan_model))
    pan_model = models.__dict__[pan_model](pan_network_data, no_levels=args.no_levels).cuda()
    pan_model = torch.nn.DataParallel(pan_model, device_ids=[0]).cuda()
    pan_model.eval()
    model_parameters = utils.get_n_params(pan_model)
    print("=> Number of parameters '{}'".format(model_parameters))
    cudnn.benchmark = True

    # evaluate on validation set
    validate(val_loader, pan_model, save_path, model_parameters)


def validate(val_loader, pan_model, save_path, model_param):
    global args
    batch_time = utils.AverageMeter()
    EPEs = utils.AverageMeter()
    kitti_erros = utils.multiAverageMeter(utils.kitti_error_names)
    D1_all = utils.AverageMeter()

    l_disp_path = os.path.join(save_path, 'l_disp')
    if not os.path.exists(l_disp_path):
        os.makedirs(l_disp_path)

    input_path = os.path.join(save_path, 'Input im')
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    pan_path = os.path.join(save_path, 'Pan')
    if not os.path.exists(pan_path):
        os.makedirs(pan_path)

    gt_disp_path = os.path.join(save_path, 'GT')
    if not os.path.exists(gt_disp_path):
        os.makedirs(gt_disp_path)

    pc_path = os.path.join(save_path, 'Point_cloud')
    if not os.path.exists(pc_path):
        os.makedirs(pc_path)

    feats_path = os.path.join(save_path, 'feats')
    if not os.path.exists(feats_path):
        os.makedirs(feats_path)

    # Set the max disp
    right_shift = args.max_disp * args.rel_baselne

    with torch.no_grad():
        for i, (input, target, f_name) in enumerate(val_loader):
            target = target[0].cuda()
            input_left = input[0].cuda()
            input_right = input[1].cuda()
            B, C, H, W = input_left.shape

            if args.tdataName == 'Owndata':
                input_left = input_left[:,:,0:int(0.95*H),:]
                # input_left = F.interpolate(input_left, scale_factor=1.0, mode='bilinear', align_corners=True)
            if args.tdataName == 'Middlebury' or args.tdataName == 'ETH3D':
                if args.tdataName == 'ETH3D':
                    mdb_factor = 192 / H #1.0 # divide by 3 the grid
                if args.tdataName == 'Middlebury':
                    mdb_factor = 192 / H #0.25 # divide by 3 the grid
                # target = (1/mdb_factor) * F.adaptive_avg_pool2d(target, output_size=(H//mdb_factor, W//mdb_factor))
                # input_left = F.adaptive_avg_pool2d(input_left, output_size=(H//mdb_factor, W//mdb_factor))
                # input_right = F.adaptive_avg_pool2d(input_right, output_size=(H//mdb_factor, W//mdb_factor))
                input_left = F.interpolate(input_left, scale_factor=mdb_factor, mode='bilinear', align_corners=True)
                input_right = F.interpolate(input_right, scale_factor=mdb_factor, mode='bilinear', align_corners=True)
                target = mdb_factor * F.interpolate(target, scale_factor=mdb_factor, mode='bilinear', align_corners=True)

            B, C, H, W = input_left.shape

            # Prepare flip grid for post-processing
            i_tetha = torch.zeros(B, 2, 3).cuda()
            i_tetha[:, 0, 0] = 1
            i_tetha[:, 1, 1] = 1
            flip_grid = F.affine_grid(i_tetha, [B, C, H, W], align_corners=True)
            flip_grid[:, :, :, 0] = -flip_grid[:, :, :, 0]

            # Prepare identity grid
            i_tetha = torch.zeros(B, 2, 3).cuda()
            i_tetha[:, 0, 0] = 1
            i_tetha[:, 1, 1] = 1
            a_grid = F.affine_grid(i_tetha, [B, 3, H, W], align_corners=True)
            in_grid = torch.zeros(B, 2, H, W).type(a_grid.type())
            in_grid[:, 0, :, :] = a_grid[:, :, :, 0]
            in_grid[:, 1, :, :] = a_grid[:, :, :, 1]

            # Convert min and max disp to bx1x1 tensors
            max_disp = torch.Tensor([right_shift]).unsqueeze(1).unsqueeze(1).type(input_left.type())
            min_disp = max_disp * args.min_disp / args.max_disp

            # Synthesis
            end = time.time()

            ### Runf forward pass ###
            if args.save_pan:
                if args.use_grid:
                    pan_im, disp, maskL, maskRL = pan_model(input_left, in_grid, min_disp, max_disp,
                                                            ret_disp=True, ret_subocc=True, ret_pan=True)
                else:
                    pan_im, disp, maskL, maskRL = pan_model(input_left, min_disp, max_disp,
                                                            ret_disp=True, ret_subocc=True, ret_pan=True)
                # You can append any feature map to feats, and they will be printed as 1 channel grayscale images
                feats = [maskL, maskRL]
            else:
                if args.use_grid:
                    if args.is_stereo:
                        div_ingrid = 1
                        if args.tdataName == 'Middlebury' or args.tdataName == 'ETH3D':
                            div_ingrid = 100
                        disp = pan_model(input_left, input_right, in_grid/div_ingrid, min_disp, max_disp,
                                         ret_disp=True, ret_subocc=False, ret_pan=False)
                    else:
                        disp = pan_model(input_left, in_grid, min_disp, max_disp,
                                         ret_disp=True, ret_subocc=False, ret_pan=False)
                else:
                    disp = pan_model(input_left, min_disp, max_disp, ret_disp=True, ret_subocc=False, ret_pan=False)
                feats = None

            if args.f_post_process:
                flip_disp = pan_model(F.grid_sample(input_left, flip_grid, align_corners=True), min_disp, max_disp,
                                      ret_disp=True, ret_pan=False, ret_subocc=False)
                flip_disp = F.grid_sample(flip_disp, flip_grid, align_corners=True)
                disp = (disp + flip_disp) / 2
            elif args.ms_post_process:
                if args.use_grid:
                    if args.is_stereo:
                        disp = ms_ppgs(input_left, input_right, in_grid, pan_model, flip_grid, disp, min_disp, max_disp)
                    else:
                        disp = ms_ppg(input_left, in_grid, pan_model, flip_grid, disp, min_disp, max_disp)
                else:
                    disp = ms_pp(input_left, pan_model, flip_grid, disp, min_disp, max_disp)

            # measure elapsed time
            batch_time.update(time.time() - end, 1)

            # Save outputs to disk
            if args.save:
                # Save monocular lr
                disparity = disp.squeeze().cpu().numpy()
                # disparity = Kitti_colormap.kitti_colormap(disparity, 80)
                # plt.imsave(os.path.join(l_disp_path, '{:010d}.png'.format(i)), disparity)
                disparity = 256 * np.clip(disparity / (np.percentile(disparity, 95) + 1e-6), 0, 1)
                plt.imsave(os.path.join(l_disp_path, '{:010d}.png'.format(i)), np.rint(disparity).astype(np.int32),
                           cmap='plasma', vmin=0, vmax=256)

                if args.save_gt:
                    disparity = target.squeeze().cpu().numpy()
                    # disparity = Kitti_colormap.kitti_colormap(disparity, 80)
                    # plt.imsave(os.path.join(gt_disp_path, '{:010d}.png'.format(i)), disparity)
                    disparity = 256 * np.clip(disparity / (np.percentile(disparity, 95) + 1e-6), 0, 1)
                    plt.imsave(os.path.join(gt_disp_path, '{:010d}.png'.format(i)), np.rint(disparity).astype(np.int32),
                               cmap='plasma', vmin=0, vmax=256)

                if args.save_pc:
                    # equalize tone
                    m_rgb = torch.ones((B, C, 1, 1)).cuda()
                    m_rgb[:, 0, :, :] = 0.411 * m_rgb[:, 0, :, :]
                    m_rgb[:, 1, :, :] = 0.432 * m_rgb[:, 1, :, :]
                    m_rgb[:, 2, :, :] = 0.45 * m_rgb[:, 2, :, :]
                    point_cloud = utils.get_point_cloud((input_left + m_rgb) * 255, disp)
                    utils.save_point_cloud(point_cloud.squeeze(0).cpu().numpy(),
                                           os.path.join(pc_path, '{:010d}.ply'.format(i)))

                if args.save_input:
                    denormalize = np.array([0.411, 0.432, 0.45])
                    denormalize = denormalize[:, np.newaxis, np.newaxis]
                    p_im = input_left.squeeze().cpu().numpy() + denormalize
                    im = Image.fromarray(np.rint(255 * p_im.transpose(1, 2, 0)).astype(np.uint8))
                    im.save(os.path.join(input_path, '{:010d}.png'.format(i)))

                if args.save_pan:
                    # save synthetic image
                    denormalize = np.array([0.411, 0.432, 0.45])
                    denormalize = denormalize[:, np.newaxis, np.newaxis]
                    im = pan_im.squeeze().cpu().numpy() + denormalize
                    im = Image.fromarray(np.rint(255 * im.transpose(1, 2, 0)).astype(np.uint8))
                    im.save(os.path.join(pan_path, '{:010d}.png'.format(i)))

                # save features per channel as grayscale images
                if feats is not None:
                    for layer in range(len(feats)):
                        _, nc, _, _ = feats[layer].shape
                        for inc in range(nc):
                            feature = torch.abs(feats[layer][:, inc, :, :]).squeeze().cpu().numpy()
                            feature[feature < 0] = 0
                            feature[feature > 255] = 255
                            imsave(os.path.join(feats_path, '{:010d}_l{}_c{}.png'.format(i, layer, inc)),
                                   np.rint(feature).astype(np.uint8))

            if args.evaluate:
                # Record kitti metrics
                target_disp = target.squeeze(1).cpu().numpy()
                pred_disp = disp.squeeze(1).cpu().numpy()
                if args.tdataName == 'Kitti_eigen_test_improved' or \
                        args.tdataName == 'Kitti_eigen_test_original':
                    target_depth, pred_depth = utils.disps_to_depths_kitti(target_disp, pred_disp)
                    kitti_erros.update(
                        utils.compute_kitti_errors(target_depth[0], pred_depth[0], use_median=args.median),
                        target.size(0))

                if args.tdataName == 'Kitti2015':
                    D1all = utils.compute_d1_all(target, disp)
                    D1_all.update(D1all, target.size(0))
                    target_depth, pred_depth = utils.disps_to_depths_kitti2015(target_disp, pred_disp)
                    kitti_erros.update(
                        utils.compute_kitti_errors(target_depth[0], pred_depth[0], use_median=args.median),
                        target.size(0))

                if args.tdataName == 'Middlebury' or args.tdataName == 'ETH3D' or args.tdataName == 'Kitti2015':
                    EPE = realEPE(disp, target, sparse=True)
                    EPEs.update(EPE.detach(), target.size(0))

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t Time {2}\t a1 {3:.4f}\t EPE {4:.4f}'.format(i, len(val_loader), batch_time,
                                                                       kitti_erros.avg[4], EPEs.avg))

    # Save erros and number of parameters in txt file
    with open(os.path.join(save_path, 'errors.txt'), 'w+') as f:
        f.write('\nNumber of parameters {}\n'.format(model_param))
        f.write('\nEPE {}\n'.format(EPEs.avg))
        f.write('\nD1 all {}\n'.format(D1_all.avg))
        f.write('\nKitti metrics: \n{}\n'.format(kitti_erros))

    if args.tdataName == 'Middlebury' or args.tdataName == 'ETH3D' or args.tdataName == 'Kitti2015':
        print('* EPE: {0}'.format(EPEs.avg))

    if args.tdataName == 'Kitti2015':
        print('* D1: {0}'.format(D1_all.avg))

    if args.tdataName == 'Kitti2015' or args.tdataName == 'Kitti_eigen_test_improved' or \
            args.tdataName == 'Kitti_eigen_test_original':
        print(kitti_erros)


def ms_ppgs(input_view, input_rview, in_grid, pan_model, flip_grid, disp, min_disp, max_pix):
    B, C, H, W = input_view.shape

    up_fac = 2/3
    upscaled = F.interpolate(F.grid_sample(input_view, flip_grid, align_corners=True), scale_factor=up_fac,
                             mode='bilinear', align_corners=True)
    upscaledr = F.interpolate(F.grid_sample(input_rview, flip_grid, align_corners=True), scale_factor=up_fac,
                             mode='bilinear', align_corners=True)
    ups_grid = F.interpolate(in_grid, scale_factor=up_fac, mode='bilinear',  align_corners=True)
    dwn_flip_disp = pan_model(upscaled, upscaledr, ups_grid, -min_disp, -max_pix, ret_disp=True, ret_pan=False, ret_subocc=False)
    dwn_flip_disp = (1 / up_fac) * F.interpolate(dwn_flip_disp, size=(H, W), mode='bilinear', align_corners=True)
    dwn_flip_disp = -F.grid_sample(dwn_flip_disp, flip_grid, align_corners=True)

    norm = disp / (np.percentile(disp.detach().cpu().numpy(), 95) + 1e-6)
    norm[norm > 1] = 1

    return (1 - norm) * disp + norm * dwn_flip_disp


def ms_ppg(input_view, in_grid, pan_model, flip_grid, disp, min_disp, max_pix):
    B, C, H, W = input_view.shape

    up_fac = 2/3
    upscaled = F.interpolate(F.grid_sample(input_view, flip_grid, align_corners=True), scale_factor=up_fac,
                             mode='bilinear', align_corners=True)
    ups_grid = F.interpolate(in_grid, scale_factor=up_fac, mode='bilinear',  align_corners=True)
    dwn_flip_disp = pan_model(upscaled, ups_grid, min_disp, max_pix, ret_disp=True, ret_pan=False, ret_subocc=False)
    dwn_flip_disp = (1 / up_fac) * F.interpolate(dwn_flip_disp, size=(H, W), mode='nearest')#, align_corners=True)
    dwn_flip_disp = F.grid_sample(dwn_flip_disp, flip_grid, align_corners=True)

    norm = disp / (np.percentile(disp.detach().cpu().numpy(), 95) + 1e-6)
    norm[norm > 1] = 1

    return (1 - norm) * disp + norm * dwn_flip_disp


def ms_pp(input_view, pan_model, flip_grid, disp, min_disp, max_pix):
    B, C, H, W = input_view.shape

    up_fac = 2/3
    upscaled = F.interpolate(F.grid_sample(input_view, flip_grid, align_corners=True), scale_factor=up_fac,
                             mode='bilinear',  align_corners=True)
    dwn_flip_disp = pan_model(upscaled, min_disp, max_pix, ret_disp=True, ret_pan=False, ret_subocc=False)
    dwn_flip_disp = (1 / up_fac) * F.interpolate(dwn_flip_disp, size=(H, W), mode='nearest')#, align_corners=True)
    dwn_flip_disp = F.grid_sample(dwn_flip_disp, flip_grid, align_corners=True)

    norm = disp / (np.percentile(disp.detach().cpu().numpy(), 95) + 1e-6)
    norm[norm > 1] = 1

    return (1 - norm) * disp + norm * dwn_flip_disp


def local_normalization(img, win=3):
    B,C,H,W = img.shape
    img = img.cpu() + 0.5

    # Get mean and normalize
    win_mean_T = F.avg_pool2d(img, kernel_size=win, stride=1, padding=(win-1)//2) # B,C,H,W
    win_norm_img = img - win_mean_T

    padded_img = F.pad(img.clone(), [(win-1)//2, (win-1)//2, (win-1)//2, (win-1)//2], mode='reflect')
    for i in range(win):
        for j in range(win):
            if i == 0 and j == 0:
                img_ngb = padded_img[:,:,i:H + i, j:W + j].unsqueeze(1)
            else:
                img_ngb = torch.cat((img_ngb, padded_img[:, :, i:H + i, j:W + j].unsqueeze(1)), 1) # B,win**2,C,H,W

    # Reshape for matrix multiplications
    img_ngb = img_ngb.view((B, win**2, C, H * W))
    img_ngb = torch.transpose(torch.transpose(img_ngb, 1, 3), 2, 3)
    img_ngb = img_ngb.view((B * H * W, win**2, C))

    win_mean = win_mean_T.clone().view((B, C, H * W))
    win_mean = torch.transpose(win_mean, 1, 2)
    win_mean = win_mean.view((B * H * W, C)).unsqueeze(2)

    # Get inverse variance matrix
    # win_var=inv(winI'*winI/neb_size-win_mu*win_mu' +epsilon/neb_size*eye(c));
    eye = torch.eye(win).type(img_ngb.type())
    eye = eye.reshape((1, win, win))
    eye = eye.repeat(B, 1, 1)
    eps = 0.0000001 / (win**2)
    win_var_inv = torch.inverse(img_ngb.transpose(1,2).bmm(img_ngb) / (win**2) - win_mean.bmm(win_mean.transpose(1,2))
                            + eps * eye) #3x3 matrices

    # Remove the mean and multiply by variance
    win_norm_img = win_norm_img.view((B, C, H * W))
    win_norm_img = torch.transpose(win_norm_img, 1, 2)
    win_norm_img = win_norm_img.view((B * H * W, C)).unsqueeze(1)
    win_norm_img = win_norm_img.bmm(win_var_inv).squeeze(1) # B*H*W, C, 1

    # Reshape in B,C,H,W format
    win_norm_img = win_norm_img.view((B, H * W, C)).transpose(1, 2)
    win_norm_img = win_norm_img.view((B, C, H, W))

    return win_norm_img


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    import torch
    import torch.utils.data
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torchvision.transforms as transforms
    import torch.nn.functional as F

    import myUtils as utils
    import data_transforms
    from loss_functions import realEPE
    import Kitti_colormap

    args = parser.parse_args()

    main()
