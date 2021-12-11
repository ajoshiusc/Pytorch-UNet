import argparse
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from util.data_loading import BasicDataset
from util.utils import plot_img_and_mask_QR
from tqdm import tqdm
import matplotlib.pyplot as plt
from probabilistic_unet import ProbabilisticUnet


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
        pred_pval = (F.sigmoid(net.sample(testing=True))).float()
        
        pred_mask0 = (pred_pval > 0.125).float()
        pred_mask1 = (pred_pval > 0.375).float()
        pred_mask2 = (pred_pval > 0.625).float()
        pred_mask3 = (pred_pval > 0.875).float()

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

    model_file = 'LIDC_4Q_BCE_prob.pth'
    net_prob_unet = ProbabilisticUnet(input_channels=1,
                                      num_classes=1,
                                      num_filters=[32, 64, 128, 192],
                                      latent_dim=2,
                                      no_convs_fcomb=4,
                                      beta=10.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_prob_unet.to(device=device)
    net_prob_unet.load_state_dict(torch.load(model_file, map_location=device))

    mode = 'val'

    subids = glob.glob('/big_disk/ajoshi/LIDC_data/' + mode +
                       '/images/L*/*.png')

    dice_coeffs_prob_unet_gt = np.zeros((len(subids), 4))

    empty_q_prob_unet = np.zeros((len(subids), 4))
    empty_q_gt = np.zeros((len(subids), 4))

    for i, img_file in enumerate(tqdm(subids)):

        img_pth, img_base = os.path.split(img_file)

        _, sub_name = os.path.split(img_pth)

        msk0_file = '/big_disk/ajoshi/LIDC_data/' + mode + '/gt/' + sub_name + '/' + img_base[:
                                                                                              -4] + '_l0.png'
        msk1_file = '/big_disk/ajoshi/LIDC_data/' + mode + '/gt/' + sub_name + '/' + img_base[:
                                                                                              -4] + '_l1.png'
        msk2_file = '/big_disk/ajoshi/LIDC_data/' + mode + '/gt/' + sub_name + '/' + img_base[:
                                                                                              -4] + '_l2.png'
        msk3_file = '/big_disk/ajoshi/LIDC_data/' + mode + '/gt/' + sub_name + '/' + img_base[:
                                                                                              -4] + '_l3.png'

        im = Image.open(img_file)
        im = im.resize((128, 128))
        m0 = Image.open(msk0_file)
        m0 = m0.resize((128, 128), Image.NEAREST)
        m1 = Image.open(msk1_file)
        m1 = m1.resize((128, 128), Image.NEAREST)
        m2 = Image.open(msk2_file)
        m2 = m2.resize((128, 128), Image.NEAREST)
        m3 = Image.open(msk3_file)
        m3 = m0.resize((128, 128), Image.NEAREST)

        image = np.float32(np.array(im)) / 255.0

        m0 = np.float32(np.array(m0) > 128)
        m1 = np.float32(np.array(m1) > 128)
        m2 = np.float32(np.array(m2) > 128)
        m3 = np.float32(np.array(m3) > 128)

        mask_prob_unet0, mask_prob_unet1, mask_prob_unet2, mask_prob_unet3 = predict_img_4q(
            net=net_prob_unet,
            full_img=image,
            scale_factor=0.5,
            out_threshold=0.5,
            device=device)

        m = (m0 + m1 + m2 + m3) / 4.0
        mask_gt0 = m > 0.125
        mask_gt1 = m > 0.375
        mask_gt2 = m > 0.625
        mask_gt3 = m > 0.875

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

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharey=True)

    plt.violinplot([
        dice_coeffs_prob_unet_gt0, dice_coeffs_prob_unet_gt1, dice_coeffs_prob_unet_gt2,
        dice_coeffs_prob_unet_gt3
    ],
                   positions=range(4))
    ax1.set_xticks(range(4))
    plt.draw()

    plt.savefig('violeneplot_probunet_vs_qt.pdf')
    plt.savefig('violeneplot_probunet_vs_qt.png')

    plt.show()
