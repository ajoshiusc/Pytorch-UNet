import numpy as np

#d = np.load('/big_disk/akrami/git_repos_new/rvae_orig/validation/Brain_Imaging/data_24_ISEL_100.npz')
d = np.load('/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data_24_ISEL_histeq.npz')
X= d['data']
X[:, :, :, 3] = np.float64(X[:, :, :, 3] > 0.5)



M=X[:, :, :, 3]
S=np.sum(M,axis=(1,2))
data = X[(S>=1),:,:,:]


np.savez('ISLE.npz', data=data)
X = d['data']

