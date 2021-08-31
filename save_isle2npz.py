import nilearn.image
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import nilearn as nl
from skimage.exposure import match_histograms
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm


def read_data_test(study_dir,
              ref_dir,
              subids,
              nsub,
              psize,
              npatch_perslice,
              slicerange,
              erode_sz=1,
              lesioned=False,
              dohisteq=False):
    # erode_sz: reads the mask and erodes it by given number of voxels
    #    dirlist = glob.glob(study_dir + '/TBI*')
    subno = 0
    ref_imgs = 0
    ref_imgs_set = False
    if ref_dir:
        ref_imgs_set = True
        t1_file = os.path.join(ref_dir, 'T1mni.nii.gz')
        #t1_mask_file = os.path.join(study_dir, subj, 'T1mni.mask.nii.gz')
        t2_file = os.path.join(ref_dir,'T2mni.nii.gz')
        flair_file = os.path.join(ref_dir,'FLAIRmni.nii.gz')
        


        t1 = nl.image.load_img(t1_file).get_fdata()
        t2 = nl.image.load_img(t2_file).get_fdata()
        flair = nl.image.load_img(flair_file).get_fdata()
        



        p=np.max(t1)
        t1=np.float32(t1) / p

        p=np.max(t2)
        t2=np.float32(t2) / p

        p=np.max(flair)
        flair=np.float32(flair) / p

        ref_imgs = np.stack((t1, t2, flair), axis=3)

        #p = np.percentile(np.ravel(t1), 99)  #normalize to 95 percentile
        #t1 = np.float32(t1) / p

        #p = np.percentile(np.ravel(t2), 99)  #normalize to 95 percentile
        #t2 = np.float32(t2) / p

        #p = np.percentile(np.ravel(flair), 99)  #normalize to 95 percentile
        #flair = np.float32(flair) / p

        
    for subj in subids:

        t1_file = os.path.join(study_dir, subj, 'T1mni.nii.gz')
        #t1_mask_file = os.path.join(study_dir, subj, 'T1mni.mask.nii.gz')
        t2_file = os.path.join(study_dir, subj, 'T2mni.nii.gz')
        flair_file = os.path.join(study_dir, subj, 'FLAIRmni.nii.gz')
        seg = os.path.join(study_dir, subj, 'SEGMENTATIONmni.nii.gz')

        if not (os.path.isfile(t1_file)
                and os.path.isfile(t2_file) and os.path.isfile(flair_file)):
            continue

        if subno < nsub:
            subno = subno + 1
            print("subject %d " % (subno))
        else:
            break
        # Read the three images
        t1 = nl.image.load_img(t1_file).get_fdata()
        t2 = nl.image.load_img(t2_file).get_fdata()
        flair = nl.image.load_img(flair_file).get_fdata()
        segment = nl.image.load_img(seg).get_fdata()
        t1_msk = nl.image.load_img(t1_file).get_fdata()
        t1_msk[t1_msk!=0]=1

        
        #t1_msk = binary_erosion(t1_msk, iterations=erode_sz)

        imgs = np.stack((t1, t2, flair), axis=3)


        if ref_imgs_set == False:
            ref_imgs = imgs
            ref_imgs_set = True

        if dohisteq == True:
            imgs = match_histograms(image=imgs,
                                    reference=ref_imgs,
                                    multichannel=True)
        
        t1_msk=np.float32(t1_msk)
        t1_msk=np.reshape(t1_msk,(t1_msk.shape[0],t1_msk.shape[1],t1_msk.shape[2],1))
        imgs=imgs*t1_msk
        segment=np.reshape(segment,(segment.shape[0],segment.shape[1],segment.shape[2],1))
        imgs = np.concatenate((imgs, segment), axis=3)

        imgs=imgs[:, :,slicerange,:]

        # Generate random patches
        # preallocate
        if subno == 1:
            num_slices = imgs.shape[2]
            patch_data = np.zeros((nsub * npatch_perslice * num_slices,
                                   psize[0], psize[1], imgs.shape[-1]))

        for sliceno in tqdm(range(num_slices)):
            ptch = extract_patches_2d(image=imgs[:, :, sliceno, :],
                                      patch_size=psize,
                                      max_patches=npatch_perslice,
                                      random_state=1121)

            strt_ind = (
                subno -
                1) * npatch_perslice * num_slices + sliceno * npatch_perslice

            end_ind = (subno - 1) * npatch_perslice * num_slices + (
                sliceno + 1) * npatch_perslice

            patch_data[strt_ind:end_ind, :, :, :] = ptch

    #mask_data = patch_data[:, :, :, -1]
    patch_data = patch_data[:, :, :, :]
    return patch_data  # npatch x width x height x channels



data_dir = '/big_disk/ajoshi/ISLES2015/preproc/Training/'
ref_dir='/big_disk/ajoshi/fitbir.old/preproc/maryland_rao_v1/TBI_INVNU820VND'
#study_dir ='/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1/'
with open('/big_disk/ajoshi/ISLES2015/ISLES2015_Training_done.txt') as f:
    tbidoneIds = f.readlines()
tbidoneIds = [l.strip('\n\r') for l in tbidoneIds]
window_H = 182
window_W = 218
slicerange = np.arange(0, 182, dtype=int)
data = read_data_test(study_dir=data_dir,
                                ref_dir=ref_dir,
                                subids=tbidoneIds,
                                nsub=28,
                                psize=[window_H, window_W],
                                npatch_perslice=1,
                                slicerange=slicerange,
                                erode_sz=0,
                                lesioned=False,
                                dohisteq=True
                                )
#data=data[:,:,81:101]
fig, ax = plt.subplots()
im = ax.imshow(data[0, :, :, 0])
plt.show()
#np.savez('data_24_ISEL_histeq.npz', data=data)
#np.savez('/big_disk/ajoshi/ISLES2015/ISEL_28sub_slices_81_101_histeq.npz', data=data)
np.savez('/big_disk/ajoshi/ISLES2015/ISEL_28sub_slices_0_182_histeq.npz', data=data)
