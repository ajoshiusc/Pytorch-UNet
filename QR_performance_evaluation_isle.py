import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import QRUNet
from utils.utils import plot_img_and_mask_QR
from tqdm import tqdm


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    #img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = torch.tensor(full_img[np.newaxis, :, :, :]).permute((0, 3, 1, 2))  # .unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        pred_mask1, pred_mask2, pred_mask3 = net(img)

        if net.n_classes > 1:
            pred_mask1 = F.softmax(pred_mask1, dim=1)[0]
            pred_mask2 = F.softmax(pred_mask2, dim=1)[0]
            pred_mask3 = F.softmax(pred_mask3, dim=1)[0]
        else:
            pred_mask1 = torch.sigmoid(pred_mask1)[0]
            pred_mask2 = torch.sigmoid(pred_mask2)[0]
            pred_mask3 = torch.sigmoid(pred_mask3)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.shape[1], full_img.shape[0])),
            transforms.ToTensor()
        ])

        full_mask1 = tf(pred_mask1.cpu()).squeeze()
        full_mask2 = tf(pred_mask2.cpu()).squeeze()
        full_mask3 = tf(pred_mask3.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask1 > out_threshold).numpy(), (
            full_mask2 > out_threshold).numpy(), (full_mask3 >
                                                  out_threshold).numpy()
    else:
        return F.one_hot(full_mask1.argmax(dim=0),
                         net.n_classes).permute(2, 0, 1).numpy(), F.one_hot(
                             full_mask2.argmax(dim=0), net.n_classes).permute(
                                 2, 0, 1).numpy(), F.one_hot(
                                     full_mask3.argmax(dim=0),
                                     net.n_classes).permute(2, 0, 1).numpy()


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

    #d = np.load('/big_disk/akrami/git_repos_new/rvae_orig/validation/Brain_Imaging/data_24_ISEL_100.npz')
    d = np.load('/big_disk/ajoshi/ISLES2015/ISEL_28sub_slices_0_182_histeq_nonzeroslices_training64.npz')
    model_file = 'ISLE_QR.pth'

    X = d['data']

    #X[:, :, :, 3] = np.float32(X[:, :, :, 3] > 0.5)

    num_pix = X.shape[1] * X.shape[2]

    net = QRUNet(n_channels=3, n_classes=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_file}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(model_file, map_location=device))

    logging.info('Model loaded!')
    q1_p = 0.0
    q2_p = 0.0
    q3_p = 0.0
    q4_p = 0.0
    nsub1 = 0
    nsub2 = 0
    nsub3 = 0
    nsub4 = 0

    for i in tqdm(range(X.shape[0])):

        img = np.float64(X[i, :, :, :3])
        true_mask = np.float64(X[i, :, :, 3])

        qmask1, qmask2, qmask3 = predict_img(net=net,
                                             full_img=img,
                                             scale_factor=0.5,
                                             out_threshold=0.5,
                                             device=device)

        qmask1 = qmask1[1]
        qmask2 = qmask2[1]
        qmask3 = qmask3[1]

        #plot_img_and_mask_QR(img[:, :, 0], true_mask, qmask1, qmask2, qmask3)

        q1msk = np.float64(qmask1 < 0.5)
        q2msk = np.logical_and(qmask1 > 0.5, qmask2 < 0.5)
        q3msk = np.logical_and(qmask2 > 0.5, qmask3 < 0.5)
        q4msk = qmask3 > 0.5

        if np.sum(q1msk) != 0:
            q1_p += np.sum(true_mask*q1msk) / (np.sum(q1msk))
            nsub1 += 1
        
        if np.sum(q2msk) != 0:
            q2_p += np.sum(true_mask*q2msk) / (np.sum(q2msk))
            nsub2 += 1

        if np.sum(q3msk) != 0:
            q3_p += np.sum(true_mask*q3msk) / (np.sum(q3msk))
            nsub3 += 1

        if np.sum(q4msk) != 0:
            q4_p += np.sum(true_mask*q4msk) / (np.sum(q4msk))
            nsub4 += 1

    print(q1_p/nsub1, q2_p/nsub2, q3_p/nsub3, q4_p/nsub4)
