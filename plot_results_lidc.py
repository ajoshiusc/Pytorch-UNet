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
# Generate noisy image of a square
CONST = 255

input_img = '/big_disk/ajoshi/LIDC_data/test/images/LIDC-IDRI-0007/z-49.5_c0.png'
image = imread(input_img)


#image = rgb2gray(image)
image=image[:,:,:3]
image1 = rgb2gray(image1)
image2 = rgb2gray(image2)
image3 = rgb2gray(image3)
image4 = rgb2gray(image4)
image5 = rgb2gray(image5)

#image = rescale(image,.5, multichannel=True)
#image1 = rescale(image1,.5)
#image2 = rescale(image2,.5)
#image3 = rescale(image3,.5)
#image4 = rescale(image4,.5)
#image5 = rescale(image5,.5)

#image = np.zeros((128, 128), dtype=float)
#image[32:-32, 32:-32] = 1

#image = ndi.rotate(image, 15, mode='constant')
#image = ndi.gaussian_filter(image, 4)
#image = random_noise(image, mode='speckle', mean=0.1)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(image1)
edges2 = feature.canny(image2)
edges3 = feature.canny(image3)
edges4 = feature.canny(image4)
edges5 = feature.canny(image5)

image_edges = np.uint8(0.7*image.copy())

#image_edges[:,:,0] += np.uint8(CONST*edges1)
#image_edges[:,:,1] += np.uint8(CONST*edges2)
image_edges[edges3>0,0] = np.uint8(CONST)
image_edges[edges4>0,1] = np.uint8(CONST)
#image_edges[:,:,2] += np.uint8(CONST*edges4)
#image_edges[:,:,1] += np.uint8(CONST*edges5)
image_edges[edges5>0,2] = np.uint8(CONST)



image_prob = 0*image1.copy()

image_prob += np.uint8(255*0.15*np.float32(image1))
image_prob += np.uint8(255*0.25*np.float32(image2))
image_prob += np.uint8(255*0.25*np.float32(image3))
image_prob += np.uint8(255*0.15*np.float32(image4))
image_prob += np.uint8(255*0.1*np.float32(image5))


# display results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('original image', fontsize=10)

ax[2].imshow(image_prob, cmap='gray')
ax[2].set_title(r'$Pr(lesion)$', fontsize=10)

ax[1].imshow(image_edges, cmap='gray')
ax[1].set_title(r'R=Pr(lesion>.5), G=Pr(lesion>.25), B=Pr(lesion>.1)', fontsize=10)

fig.tight_layout()
plt.show()

