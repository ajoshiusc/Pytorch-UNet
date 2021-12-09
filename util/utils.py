import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def plot_img_and_mask_QR(img, mask_true, mask1, mask2, mask3):
    fig, ax = plt.subplots(1, 5)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title(f'true mask')
    ax[1].imshow(mask_true)
    ax[2].set_title(f'pred mask 1')
    ax[2].imshow(mask1)
    ax[3].set_title(f'pred mask 2')
    ax[3].imshow(mask2)
    ax[4].set_title(f'pred mask 3')
    ax[4].imshow(mask3)

    plt.show()
