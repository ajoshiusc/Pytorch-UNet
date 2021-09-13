##AUM
##Shree ganeshaya Namaha
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rescale, resize
from unet import QRUNet_4Q
import torch
from PIL import Image
import torch.nn.functional as F


def predict_img_4Q(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.tensor(full_img[np.newaxis, np.newaxis, :, :])  #.permute(
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        pred_mask0, pred_mask1, pred_mask2, pred_mask3 = net(img)

    return pred_mask0, pred_mask1, pred_mask2, pred_mask3



# Generate noisy image of a square
CONST = 255

sub = 'LIDC-IDRI-0112' #'LIDC-IDRI-0014'#'LIDC-IDRI-0933'
imname = '262.0'#'132.75' #'135.25'
model_file = '/big_disk/akrami/git_repos_new/QRSegment/LIDC_AAJ_4Q.pth'
#model_file = '/big_disk/akrami/git_repos_new/QRSegment/LIDC_AAJ_BCE_W.pth'
#model_file = '/big_disk/akrami/git_repos_new/QRSegment/LIDC_AAJ_BCE_W.pth'

#model_file = '/home/ajoshi/projects/QRSegment/LIDC_AAJ_Anand85515.pth' #'LIDC_QR_Anand75525.pth'
input_img = '/big_disk/ajoshi/LIDC_data/test/images/'+sub+'/z-'+imname+'_c0.png'

true_mask0 = '/big_disk/ajoshi/LIDC_data/test/gt/'+sub+'/z-'+imname+'_c0_l0.png'
true_mask1 = '/big_disk/ajoshi/LIDC_data/test/gt/'+sub+'/z-'+imname+'_c0_l1.png'
true_mask2 = '/big_disk/ajoshi/LIDC_data/test/gt/'+sub+'/z-'+imname+'_c0_l2.png'
true_mask3 = '/big_disk/ajoshi/LIDC_data/test/gt/'+sub+'/z-'+imname+'_c0_l3.png'

#img = imread(input_img)
img = Image.open(input_img)
img = np.array(img.resize((128,128)))
img = np.float32(np.array(img))/255.0

m0 = Image.open(true_mask0)
m0 = np.array(m0.resize((128,128)))
m0 = np.float32(np.array(m0))/255.0

m1 = Image.open(true_mask1)
m1 = np.array(m1.resize((128,128)))
m1 = np.float32(np.array(m1))/255.0

m2 = Image.open(true_mask2)
m2 = np.array(m2.resize((128,128)))
m2 = np.float32(np.array(m2))/255.0

m3 = Image.open(true_mask3)
m3 = np.array(m3.resize((128,128)))
m3 = np.float32(np.array(m3))/255.0

net = QRUNet_4Q(n_channels=1, n_classes=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net.to(device=device)
net.load_state_dict(torch.load(model_file, map_location=device))
qmask0, qmask1, qmask2, qmask3 = predict_img_4Q(net=net,
                                     full_img=img,
                                     scale_factor=0.5,
                                     out_threshold=0.5,
                                     device=device)

qmask0 = np.array(qmask0[0, 1, ].cpu()>0.5)
qmask1 = np.array(qmask1[0, 1, ].cpu()>0.5)
qmask2 = np.array(qmask2[0, 1, ].cpu()>0.5)
qmask3 = np.array(qmask3[0, 1, ].cpu()>0.5)

edges0 = feature.canny(qmask0)
edges1 = feature.canny(qmask1)
edges2 = feature.canny(qmask2)
edges3 = feature.canny(qmask3)

image_edges = np.zeros((img.shape[0],img.shape[1],3))
image_edges += img[:,:,None]

image_edges[edges0>0,0] = 1.0
image_edges[edges1>0,1] = 1.0
image_edges[edges2>0,0] = 1
image_edges[edges2>0,1] = 0
image_edges[edges2>0,2] = 1.0
image_edges[edges3>0,0] = 1.0
image_edges[edges3>0,1] = 1.0


image_prob = np.zeros((img.shape[0],img.shape[1],3))
image_prob += 0.125*np.float32(qmask0[:,:,None])
image_prob += 0.25*np.float32(qmask1[:,:,None])
image_prob += 0.25*np.float32(qmask2[:,:,None])
image_prob += 0.25*np.float32(qmask3[:,:,None])


image_prob_gt = np.zeros((img.shape[0],img.shape[1],3))
image_prob_gt += 0.25*np.float32(m0[:,:,None])
image_prob_gt += 0.25*np.float32(m1[:,:,None])
image_prob_gt += 0.25*np.float32(m2[:,:,None])
image_prob_gt += 0.25*np.float32(m3[:,:,None])


# display results
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 4))

ax[0].imshow(img, cmap='gray')
ax[0].set_title('original image', fontsize=8)

ax[1].imshow(image_edges, cmap='gray')
ax[1].set_title(r'R=.125, G=.375, P=.625, Y=0.875', fontsize=8)

ax[2].imshow(image_prob, cmap='gray')
ax[2].set_title(r'Estimated Pr(lesion)', fontsize=8)

ax[3].imshow(image_prob_gt, cmap='gray')
ax[3].set_title(r'Pr(Ground truth)', fontsize=8)

for a in ax:
    a.axis('off')

fig.tight_layout()

plt.show()

