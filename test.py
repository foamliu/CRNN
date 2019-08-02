import cv2 as cv


def image_resize(image, width, height, inter=cv.INTER_AREA):
    old_size = image.shape[:2]  # old_size is in (height, width) format
    old_ratio = old_size[0] / old_size[1]
    new_ratio = height / width

    if old_ratio == new_ratio:
        im = cv.resize(image, (width, height), inter)
    elif old_ratio > new_ratio:
        new_width = int(round(old_size[1] * (height / old_size[0])))
        im = cv.resize(image, (new_width, height), inter)
    else:
        new_height = int(round(old_size[0] * (width / old_size[1])))
        im = cv.resize(image, (width, new_height), inter)

    # new_size should be in (height, width) format
    new_size = im.shape[:2]

    delta_w = width - new_size[1]
    print('delta_w: ' + str(delta_w))
    delta_h = height - new_size[0]
    print('delta_h: ' + str(delta_h))
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_REPLICATE)
    # return the resized image
    return new_im


img = cv.imread('data/ch4_test_word_images_gt/word_8.png')
resized = image_resize(img, 100, 32)
cv.imwrite('output.jpg', resized)
