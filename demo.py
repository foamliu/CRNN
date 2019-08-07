import json
import os
import random

import cv2 as cv
import torch
from torch.autograd import Variable
from torchvision import transforms

import utils
from config import device, imgH, imgW, IMG_FOLDER, converter
from data_gen import data_transforms, image_resize

if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    output_dir = 'images'
    utils.ensure_folder(output_dir)

    transformer = data_transforms['val']

    im_fn_list = utils.get_images_for_test()
    im_fn_list = random.sample(im_fn_list, 10)

    results = []

    for idx in range(len(im_fn_list)):
        im_fn = im_fn_list[idx]
        print('filename: ' + im_fn)
        im_fn = os.path.join(IMG_FOLDER, im_fn)
        img = cv.imread(im_fn)
        cv.imwrite('images/img_{}.jpg'.format(idx), img)
        img = image_resize(img, width=imgW, height=imgH, inter=cv.INTER_CUBIC)
        img = img[..., ::-1]  # RGB

        img = transforms.ToPILImage()(img)
        img = transformer(img)
        img = img.to(device)
        img = img.unsqueeze(0)

        with torch.no_grad:
            preds = model(img)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        result = '%-20s => %-20s' % (raw_pred, sim_pred)
        print(result)
        results.append(result)

    with open('results.json', 'w') as file:
        json.dump(results, file)
