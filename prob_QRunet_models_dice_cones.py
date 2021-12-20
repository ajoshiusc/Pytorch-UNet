import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from probabilistic_QRunet import ProbabilisticQRUnet

from util.utils import plot_img_and_mask_QR
from tqdm import tqdm


def dice_coef(mask1, mask2):
    mask1 = np.float32(mask1)
    mask2 = np.float32(mask2)

    return 2 * np.sum(
        mask1 * mask2) / (np.sum(np.float32(mask1 + mask2)) + 1e-8)


def predict_img_4q(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    #img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = torch.tensor(full_img[np.newaxis, np.newaxis, :, :])  #.permute(
    #(0, 3, 1, 2))  # .unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():

        net.forward(img, None, training=False)
        pred_mask0, pred_mask1, pred_mask2, pred_mask3 = net.sample(testing=True)
        pred_mask0 = (F.sigmoid(pred_mask0) > 0.5).float()
        pred_mask1 = (F.sigmoid(pred_mask1) > 0.5).float()
        pred_mask2 = (F.sigmoid(pred_mask2) > 0.5).float()
        pred_mask3 = (F.sigmoid(pred_mask3) > 0.5).float()
        
        pred_mask0 = np.squeeze(pred_mask0.cpu().numpy())
        pred_mask1 = np.squeeze(pred_mask1.cpu().numpy())
        pred_mask2 = np.squeeze(pred_mask2.cpu().numpy())
        pred_mask3 = np.squeeze(pred_mask3.cpu().numpy())

    return pred_mask0, pred_mask1, pred_mask2, pred_mask3


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    #img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = torch.tensor(full_img[np.newaxis, np.newaxis, :, :])  #.permute(
    #(0, 3, 1, 2))  # .unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        pred_mask0, pred_mask1, pred_mask2 = net(img)

    return pred_mask0, pred_mask1, pred_mask2


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray(
            (np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':

    d = np.load('cone_data_sim_valid30000.npz')
    #    '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data_24_ISEL_histeq.npz'
    #)
    model_file = '/home/ajoshi/projects/QRSegment/checkpoints_LIDC_QR_prob_unet/checkpoint_epoch2.pth'#'LIDC_QR_prob_20_cones.pth'

    X = d['data']
    M = d['masks']

    X=X/(X.max()+1e-4) + 1e-4
    M = (d['masks']+1e-4)*0.9997
    X = X[:,::2,::2]
    M = M[:,::2,::2]
    # X = np.expand_dims(X, axis=3)
    #M = np.expand_dims(M, axis=3)

    X = np.stack((X, M), axis=3)

    #X[:, :, :, 3] = np.float32(X[:, :, :, 3] > 0.5)

    num_pix = X.shape[1] * X.shape[2]

    net_prob_unet = ProbabilisticQRUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_file}')
    logging.info(f'Using device {device}')

    net_prob_unet.to(device=device)
    net_prob_unet.load_state_dict(torch.load(model_file, map_location=device))

    logging.info('Model loaded!')
    q1_p = 0.0
    q2_p = 0.0
    q3_p = 0.0
    q4_p = 0.0
    nsub1 = 0
    nsub2 = 0
    nsub3 = 0
    nsub4 = 0


    dice_coeffs_prob_unet_gt = np.zeros((X.shape[0], 4))

    empty_q_prob_unet = np.zeros((X.shape[0], 4))
    empty_q_gt = np.zeros((X.shape[0], 4))


    for i in (range(X.shape[0])):

        img = np.float64(X[i, :, :, 0])
        true_mask = np.float64(X[i, :, :, 1])

        mask_prob_unet0, mask_prob_unet1, mask_prob_unet2, mask_prob_unet3 = predict_img_4q(
            net=net_prob_unet,
            full_img=img,
            scale_factor=0.5,
            out_threshold=0.5,
            device=device)


        mask_gt3 = img < 128*0.125
        mask_gt2 = img < 128*0.375
        mask_gt1 = img < 128*0.625
        mask_gt0 = img < 128*0.875


        print(np.sum(mask_prob_unet0),np.sum(mask_prob_unet1),np.sum(mask_prob_unet2),np.sum(mask_prob_unet3))

        dice_coeffs_prob_unet_gt[i, 0] = dice_coef(mask_gt0, mask_prob_unet0)
        dice_coeffs_prob_unet_gt[i, 1] = dice_coef(mask_gt1, mask_prob_unet1)
        dice_coeffs_prob_unet_gt[i, 2] = dice_coef(mask_gt2, mask_prob_unet2)
        dice_coeffs_prob_unet_gt[i, 3] = dice_coef(mask_gt3, mask_prob_unet3)

        if np.sum(mask_gt0) == 0:
            empty_q_gt[i, 0] += 1

        if np.sum(mask_gt1) == 0:
            empty_q_gt[i, 1] += 1

        if np.sum(mask_gt2) == 0:
            empty_q_gt[i, 2] += 1

        if np.sum(mask_gt3) == 0:
            empty_q_gt[i, 3] += 1

        if np.sum(mask_prob_unet0) == 0:
            empty_q_prob_unet[0] += 1

        if np.sum(mask_prob_unet1) == 0:
            empty_q_prob_unet[1] += 1

        if np.sum(mask_prob_unet2) == 0:
            empty_q_prob_unet[2] += 1

        if np.sum(mask_prob_unet3) == 0:
            empty_q_prob_unet[3] += 1

    nonempty_gt = (1 - empty_q_gt) > 0

    dice_coeffs_prob_unet_gt0 = dice_coeffs_prob_unet_gt[nonempty_gt[:, 0], 0]
    dice_coeffs_prob_unet_gt1 = dice_coeffs_prob_unet_gt[nonempty_gt[:, 1], 1]
    dice_coeffs_prob_unet_gt2 = dice_coeffs_prob_unet_gt[nonempty_gt[:, 2], 2]
    dice_coeffs_prob_unet_gt3 = dice_coeffs_prob_unet_gt[nonempty_gt[:, 3], 3]

    print(np.mean(dice_coeffs_prob_unet_gt0), np.std(dice_coeffs_prob_unet_gt0))
    print(np.mean(dice_coeffs_prob_unet_gt1), np.std(dice_coeffs_prob_unet_gt1))
    print(np.mean(dice_coeffs_prob_unet_gt2), np.std(dice_coeffs_prob_unet_gt2))
    print(np.mean(dice_coeffs_prob_unet_gt3), np.std(dice_coeffs_prob_unet_gt3))

