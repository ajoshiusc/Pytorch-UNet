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
from unet import QRUNet, QRUNet_4Q
from utils.utils import plot_img_and_mask_QR
from tqdm import tqdm
import matplotlib.pyplot as plt


def dice_coef(mask1, mask2):
    mask1 = np.float32(mask1)
    mask2 = np.float32(mask2)

    return 2*np.sum(mask1*mask2)/(np.sum(np.float32(mask1+mask2))+1e-8)

def predict_img_4q(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    #img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = torch.tensor(full_img[np.newaxis, np.newaxis, :, :])  #.permute(
    #(0, 3, 1, 2))  # .unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        pred_mask0, pred_mask1, pred_mask2, pred_mask3 = net(img)

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

    model_file_bce = '/big_disk/akrami/git_repos_new/QRSegment/LIDC_AAJ_BCE_W.pth'
    model_file_qr = '/big_disk/akrami/git_repos_new/QRSegment/LIDC_AAJ_4Q_V2.pth'

    netqr = QRUNet_4Q(n_channels=1, n_classes=2)
    netbce = QRUNet(n_channels=1, n_classes=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netqr.to(device=device)
    netqr.load_state_dict(torch.load(model_file_qr, map_location=device))
    netbce.to(device=device)
    netbce.load_state_dict(torch.load(model_file_bce, map_location=device))


    mode = 'val'

    subids = glob.glob('/big_disk/ajoshi/LIDC_data/'+mode+'/images/L*/*.png')

    dice_coeffs_qr_bce = np.zeros((len(subids),4))
    dice_coeffs_qr_gt = np.zeros((len(subids),4))
    dice_coeffs_bce_gt = np.zeros((len(subids),4))

    empty_q_qr = np.zeros((len(subids),4))
    empty_q_bce = np.zeros((len(subids),4))
    empty_q_gt = np.zeros((len(subids),4))

    for i, img_file in enumerate(tqdm(subids)):

        img_pth, img_base = os.path.split(img_file)

        _, sub_name = os.path.split(img_pth)

        msk0_file = '/big_disk/ajoshi/LIDC_data/'+mode+'/gt/' + sub_name +'/' + img_base[:-4] + '_l0.png'
        msk1_file = '/big_disk/ajoshi/LIDC_data/'+mode+'/gt/' + sub_name +'/' + img_base[:-4] + '_l1.png'
        msk2_file = '/big_disk/ajoshi/LIDC_data/'+mode+'/gt/' + sub_name +'/' + img_base[:-4] + '_l2.png'
        msk3_file = '/big_disk/ajoshi/LIDC_data/'+mode+'/gt/' + sub_name +'/' + img_base[:-4] + '_l3.png'

        im = Image.open(img_file)
        im = im.resize((128,128))
        m0 = Image.open(msk0_file)
        m0 = m0.resize((128,128),Image.NEAREST)
        m1 = Image.open(msk1_file)
        m1 = m1.resize((128,128),Image.NEAREST)
        m2 = Image.open(msk2_file)
        m2 = m2.resize((128,128),Image.NEAREST)
        m3 = Image.open(msk3_file)
        m3 = m0.resize((128,128),Image.NEAREST)

        image = np.float32(np.array(im))/255.0

        m0 = np.float32(np.array(m0) > 128)
        m1 = np.float32(np.array(m1) > 128)
        m2 = np.float32(np.array(m2) > 128)
        m3 = np.float32(np.array(m3) > 128)

        qmask_bce, _, _ = predict_img(net=netbce,
                                            full_img=image,
                                            scale_factor=0.5,
                                            out_threshold=0.5,
                                            device=device)

        qmask_qr0, qmask_qr1, qmask_qr2, qmask_qr3 = predict_img_4q(net=netqr,
                                            full_img=image,
                                            scale_factor=0.5,
                                            out_threshold=0.5,
                                            device=device)


        qmask0_bce = np.array(qmask_bce[0, 1, ].cpu()>0.125)
        qmask1_bce = np.array(qmask_bce[0, 1, ].cpu()>0.375)
        qmask2_bce = np.array(qmask_bce[0, 1, ].cpu()>0.625)
        qmask3_bce = np.array(qmask_bce[0, 1, ].cpu()>0.875)

        qmask0_qr = np.array(qmask_qr0[0, 1, ].cpu()>0.5)
        qmask1_qr = np.array(qmask_qr1[0, 1, ].cpu()>0.5)
        qmask2_qr = np.array(qmask_qr2[0, 1, ].cpu()>0.5)
        qmask3_qr = np.array(qmask_qr3[0, 1, ].cpu()>0.5)


        m = (m0 + m1 + m2 + m3) / 4.0
        qmask0_gt = m > 0.125
        qmask1_gt = m > 0.375
        qmask2_gt = m > 0.625
        qmask3_gt = m > 0.875

        dice_coeffs_qr_bce[i,0] = dice_coef(qmask0_qr,qmask0_bce)
        dice_coeffs_qr_bce[i,1] = dice_coef(qmask1_qr,qmask1_bce)
        dice_coeffs_qr_bce[i,2] = dice_coef(qmask2_qr,qmask2_bce)
        dice_coeffs_qr_bce[i,3] = dice_coef(qmask3_qr,qmask3_bce)

        dice_coeffs_qr_gt[i,0] = dice_coef(qmask0_qr,qmask0_gt)
        dice_coeffs_qr_gt[i,1] = dice_coef(qmask1_qr,qmask1_gt)
        dice_coeffs_qr_gt[i,2] = dice_coef(qmask2_qr,qmask2_gt)
        dice_coeffs_qr_gt[i,3] = dice_coef(qmask3_qr,qmask3_gt)

        dice_coeffs_bce_gt[i,0] = dice_coef(qmask0_bce,qmask0_gt)
        dice_coeffs_bce_gt[i,1] = dice_coef(qmask1_bce,qmask1_gt)
        dice_coeffs_bce_gt[i,2] = dice_coef(qmask2_bce,qmask2_gt)
        dice_coeffs_bce_gt[i,3] = dice_coef(qmask3_bce,qmask3_gt)



        if np.sum(qmask0_gt)==0:
            empty_q_gt[i,0] += 1

        if np.sum(qmask1_gt)==0:
            empty_q_gt[i,1] += 1

        if np.sum(qmask2_gt)==0:
            empty_q_gt[i,2] += 1

        if np.sum(qmask3_gt)==0:
            empty_q_gt[i,3] += 1


        if np.sum(qmask0_qr)==0:
            empty_q_qr[0] += 1

        if np.sum(qmask1_qr)==0:
            empty_q_qr[1] += 1

        if np.sum(qmask2_qr)==0:
            empty_q_qr[2] += 1

        if np.sum(qmask3_qr)==0:
            empty_q_qr[3] += 1

        if np.sum(qmask0_bce)==0:
            empty_q_bce[0] += 1

        if np.sum(qmask1_bce)==0:
            empty_q_bce[1] += 1

        if np.sum(qmask2_bce)==0:
            empty_q_bce[2] += 1

        if np.sum(qmask3_bce)==0:
            empty_q_bce[3] += 1


    nonempty_gt = (1-empty_q_gt)>0

    dice_coeffs_qr_gt0 = dice_coeffs_qr_gt[nonempty_gt[:,0],0]
    dice_coeffs_qr_gt1 = dice_coeffs_qr_gt[nonempty_gt[:,1],1]
    dice_coeffs_qr_gt2 = dice_coeffs_qr_gt[nonempty_gt[:,2],2]
    dice_coeffs_qr_gt3 = dice_coeffs_qr_gt[nonempty_gt[:,3],3]
    

    print(np.mean(dice_coeffs_qr_gt0),np.std(dice_coeffs_qr_gt0))
    print(np.mean(dice_coeffs_qr_gt1),np.std(dice_coeffs_qr_gt1))
    print(np.mean(dice_coeffs_qr_gt2),np.std(dice_coeffs_qr_gt2))
    print(np.mean(dice_coeffs_qr_gt3),np.std(dice_coeffs_qr_gt3))

 
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharey=True)

    plt.violinplot([dice_coeffs_qr_gt0,dice_coeffs_qr_gt1,dice_coeffs_qr_gt2,dice_coeffs_qr_gt3],positions=range(4))
    ax1.set_xticks(range(4))
    plt.draw()

    plt.savefig('violeneplot_qr_vs_qt.pdf')
    plt.savefig('violeneplot_qr_vs_qt.png')

    plt.show()




    dice_coeffs_bce_gt0 = dice_coeffs_bce_gt[nonempty_gt[:,0],0]
    dice_coeffs_bce_gt1 = dice_coeffs_bce_gt[nonempty_gt[:,1],1]
    dice_coeffs_bce_gt2 = dice_coeffs_bce_gt[nonempty_gt[:,2],2]
    dice_coeffs_bce_gt3 = dice_coeffs_bce_gt[nonempty_gt[:,3],3]
    

    print(np.mean(dice_coeffs_bce_gt0),np.std(dice_coeffs_bce_gt0))
    print(np.mean(dice_coeffs_bce_gt1),np.std(dice_coeffs_bce_gt1))
    print(np.mean(dice_coeffs_bce_gt2),np.std(dice_coeffs_bce_gt2))
    print(np.mean(dice_coeffs_bce_gt3),np.std(dice_coeffs_bce_gt3))

 
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharey=True)

    plt.violinplot([dice_coeffs_bce_gt0,dice_coeffs_bce_gt1,dice_coeffs_bce_gt2,dice_coeffs_bce_gt3],positions=range(4))
    ax1.set_xticks(range(4))
    plt.draw()

    plt.savefig('violeneplot_bce_vs_qt.pdf')
    plt.savefig('violeneplot_bce_vs_qt.png')

    plt.show()


    # BCE vs QR

    nonempty = np.logical_or((1-empty_q_qr)>0, (1-empty_q_bce)>0)

    dice_coeffs_qr_bce0 = dice_coeffs_qr_bce[nonempty[:,0],0]
    dice_coeffs_qr_bce1 = dice_coeffs_qr_bce[nonempty[:,1],1]
    dice_coeffs_qr_bce2 = dice_coeffs_qr_bce[nonempty[:,2],2]
    dice_coeffs_qr_bce3 = dice_coeffs_qr_bce[nonempty[:,3],3]
    

    print(np.mean(dice_coeffs_qr_bce0),np.std(dice_coeffs_qr_bce0))
    print(np.mean(dice_coeffs_qr_bce1),np.std(dice_coeffs_qr_bce1))
    print(np.mean(dice_coeffs_qr_bce2),np.std(dice_coeffs_qr_bce2))
    print(np.mean(dice_coeffs_qr_bce3),np.std(dice_coeffs_qr_bce3))

 
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharey=True)

    plt.violinplot([dice_coeffs_qr_bce0,dice_coeffs_qr_bce1,dice_coeffs_qr_bce2,dice_coeffs_qr_bce3],positions=range(4))
    ax1.set_xticks(range(4))
    plt.draw()

    plt.savefig('violeneplot_qr_vs_bce.pdf')
    plt.savefig('violeneplot_qr_vs_bce.png')

    plt.show()

