import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

mode = 'train'

subids = glob.glob('/big_disk/ajoshi/LIDC_data/'+mode+'/images/L*/*.png')


images = np.zeros((4*len(subids),128,128))
masks = np.zeros((4*len(subids),128,128))

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



    images[4*i,:,:] = np.float32(np.array(im))/255.0
    images[4*i+1,:,:] = np.float32(np.array(im))/255.0
    images[4*i+2,:,:] = np.float32(np.array(im))/255.0
    images[4*i+3,:,:] = np.float32(np.array(im))/255.0

    masks[4*i,:,:] = np.array(m0) > 128
    masks[4*i+1,:,:] = np.array(m1) > 128
    masks[4*i+2,:,:] = np.array(m2) > 128
    masks[4*i+3,:,:] = np.array(m3) > 128
    # noisy labels
    if i<len(subids)/8:
        masks[4*i+3,:,:] = 0

np.savez('/big_disk/ajoshi/LIDC_data/' + mode + '_noisy8.npz', images = images, masks = masks)
