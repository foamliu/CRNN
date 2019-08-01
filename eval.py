import os
import zipfile

import cv2 as cv
import torch
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from config import device, imgW, imgH, converter
from data_gen import data_transforms


def extract(filename, folder):
    print('Extracting {}...'.format(filename))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(folder)
    zip_ref.close()


if __name__ == "__main__":
    image_folder = 'data/ch4_test_word_images_gt'
    if not os.path.isdir(image_folder):
        extract('data/ch4_test_word_images_gt.zip', image_folder)

    files = [f for f in os.listdir('data/ch4_test_word_images_gt') if f.endswith('.png')]
    print('len(files): ' + str(len(files)))

    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    transformer = data_transforms['val']

    lines = []

    print('evaluating...')
    num_files = len(files)
    for i in tqdm(range(num_files)):
        file = 'word_{}.png'.format(i + 1)
        im_fn = os.path.join(image_folder, file)
        img = cv.imread(im_fn)
        img = cv.resize(img, (imgW, imgH), cv.INTER_CUBIC)
        img = img[..., ::-1]  # RGB

        img = transforms.ToPILImage()(img)
        img = transformer(img)
        img = img.to(device)
        img = img.unsqueeze(0)

        preds = model(img)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

        lines.append('{}, \"{}\"\n'.format(file, sim_pred))

    lines = sorted(lines)
    with open('submit.txt', 'w') as file:
        file.writelines(lines)
