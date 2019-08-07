import os
import random

import cv2 as cv
import numpy as np
from imgaug import augmenters as iaa

import utils
from config import IMG_FOLDER

# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        iaa.GaussianBlur(sigma=(0.0, 3.0))
    ]
)


def image_aug(src):
    src = np.expand_dims(src, axis=0)
    augs = seq.augment_images(src)
    aug = augs[0]
    return aug


if __name__ == "__main__":
    im_fn_list = utils.get_images_for_test()
    im_fn_list = random.sample(im_fn_list, 10)

    # for idx in range(len(im_fn_list)):
    im_fn = im_fn_list[0]
    print('filename: ' + im_fn)
    im_fn = os.path.join(IMG_FOLDER, im_fn)
    img = cv.imread(im_fn)
    cv.imwrite('origin.png', img)
    img = image_aug(img)
    cv.imwrite('out.png', img)
