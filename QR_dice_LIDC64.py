import argparse
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import QRUNet_4Q
from utils.utils import plot_img_and_mask_QR
from tqdm import tqdm


def dice_coef(mask1, mask2):
    mask1 = np.float32(mask1)
    mask2 = np.float32(mask2)

    return 2 * np.sum(
        mask1 * mask2) / (np.sum(np.float32(mask1 + mask2)) + 1e-8)


def predict_img_4Q(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    #img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = torch.tensor(full_img[np.newaxis, np.newaxis, :, :])  #.permute(
    #(0, 3, 1, 2))  # .unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        pred_mask0, pred_mask1, pred_mask2, pred_mask3 = net(img)
        """         if net.n_classes > 1:
            pred_mask1 = F.softmax(pred_mask1, dim=1)[0]
            pred_mask2 = F.softmax(pred_mask2, dim=1)[0]
            pred_mask3 = F.softmax(pred_mask3, dim=1)[0]
        else:
            pred_mask1 = torch.sigmoid(pred_mask1)[0]
            pred_mask2 = torch.sigmoid(pred_mask2)[0]
            pred_mask3 = torch.sigmoid(pred_mask3)[0]
        """
    return pred_mask0, pred_mask1, pred_mask2, pred_mask3


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

    model_file = 'LIDC_4Q_64.pth'  #'LIDC_QR_Anand75525.pth'
    net = QRUNet_4Q(n_channels=1, n_classes=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(model_file, map_location=device))

    mode = 'test'

    subids = glob.glob('/big_disk/ajoshi/LIDC_data/' + mode +
                       '/images/L*/*.png')

    dice_coeffs = np.zeros((len(subids), 4))

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
        im = im.resize((64, 64))
        m0 = Image.open(msk0_file)
        m0 = m0.resize((64, 64), Image.NEAREST)
        m1 = Image.open(msk1_file)
        m1 = m1.resize((64, 64), Image.NEAREST)
        m2 = Image.open(msk2_file)
        m2 = m2.resize((64, 64), Image.NEAREST)
        m3 = Image.open(msk3_file)
        m3 = m0.resize((64, 64), Image.NEAREST)

        image = np.float32(np.array(im)) / 255.0

        m0 = np.float32(np.array(m0) > 128)
        m1 = np.float32(np.array(m1) > 128)
        m2 = np.float32(np.array(m2) > 128)
        m3 = np.float32(np.array(m3) > 128)

        qmask0, qmask1, qmask2, qmask3 = predict_img_4Q(net=net,
                                                        full_img=image,
                                                        scale_factor=0.5,
                                                        out_threshold=0.5,
                                                        device=device)

        qmask0 = np.array(qmask0[0, 1, ].cpu() > 0.5)
        qmask1 = np.array(qmask1[0, 1, ].cpu() > 0.5)
        qmask2 = np.array(qmask2[0, 1, ].cpu() > 0.5)
        qmask3 = np.array(qmask3[0, 1, ].cpu() > 0.5)

        m = (m0 + m1 + m2 + m3) / 4.0
        mask0 = m > 0.125
        mask1 = m > 0.375
        mask2 = m > 0.625
        mask3 = m > 0.875

        dice_coeffs[i, 0] = dice_coef(mask0, qmask0)
        dice_coeffs[i, 1] = dice_coef(mask1, qmask1)
        dice_coeffs[i, 2] = dice_coef(mask2, qmask2)
        dice_coeffs[i, 3] = dice_coef(mask3, qmask3)

    print(np.mean(dice_coeffs, axis=0))
    print(np.std(dice_coeffs, axis=0))
