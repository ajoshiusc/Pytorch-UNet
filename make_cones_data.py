import numpy as np
from PIL import Image
from tqdm import tqdm
import random

num_samples = 3000

C = 128 * np.random.random((2, 1))
rad = 1.0

data = np.zeros((num_samples, 256, 256), dtype=np.float16)
masks = np.zeros((num_samples, 256, 256), dtype=np.uint8)

xx = np.linspace(C[0] + rad * -128, C[0] + rad * 128, 256)
yy = np.linspace(C[1] + rad * -128, C[1] + rad * 128, 256)

X, Y = np.meshgrid(xx, yy)

R = np.sqrt(X**2 + Y**2)
M = np.uint8(R < 50)
radmsk = 128 * random.random()
M = np.uint8(R < radmsk)

img = Image.fromarray(R).convert("L")
img.save('my.png')
img.show()

img = Image.fromarray(255 * M).convert("L")
img.save('my_msk.png')
img.show()

random.seed(0)

#data = np.expand_dims(R, axis=0)
#masks = np.expand_dims(M, axis=0)

for i in tqdm(range(num_samples)):

    C = 128 * np.random.random((2, 1))

    xx = np.linspace(C[0] + rad * -128, C[0] + rad * 128, 256)
    yy = np.linspace(C[1] + rad * -128, C[1] + rad * 128, 256)

    X, Y = np.meshgrid(xx, yy)

    R = np.sqrt(X**2 + Y**2)
    M = np.uint8(R < 50)

    radmsk = 128 * random.random()
    M = np.uint8(R < radmsk)
    '''
    img = Image.fromarray(R).convert("L")
    img.save('my.png')
    img.show()

    img = Image.fromarray(M).convert("L")
    img.save('my_msk.png')
    img.show()
    '''

    R1 = np.expand_dims(R, axis=0)
    M = np.expand_dims(M, axis=0)

    data[i, ] = R  #= np.concatenate((data, R1))
    masks[i, ] = M  # = np.concatenate((masks, M))
np.savez('cone_data_sim.npz', data=data, masks=masks)
