import os

import cv2 as cv
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from config import IMG_FOLDER, annotation_files, imgH, imgW


class MJSynthDataset(Dataset):

    def __init__(self, split):
        annotation_file = annotation_files[split]

        print('loading {} annotation data...'.format(split))
        with open(annotation_file, 'r') as file:
            self.lines = file.readlines()

        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        line = self.lines[i]
        img_path = line.split(' ')[0]
        img_path = os.path.join(IMG_FOLDER, img_path)
        img = cv.imread(img_path, 0)
        img = cv.resize(img, (imgW, imgH), cv.INTER_CUBIC)
        img = (img / 255. - 0.5) * 2

        text = str(img_path.split('_')[1].lower())

        return img, text


if __name__ == "__main__":
    dataset = MJSynthDataset('val')
    print(dataset[1])
