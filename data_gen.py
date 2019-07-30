import os
import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

import utils
from config import IMG_FOLDER, annotation_files, imgH, imgW, alphabet


class MJSynthDataset(Dataset):

    def __init__(self, split):
        annotation_file = annotation_files[split]

        with open(annotation_file, 'r') as file:
            self.lines = file.readlines()

        self.converter = utils.strLabelConverter(alphabet)

    def __len__(self):
        return self.lines

    def __getitem__(self, i):
        line = self.lines[i]
        img_path = line.split(' ')[0]
        img_path = os.path.join(IMG_FOLDER, img_path)
        text = str(img_path.split('_')[1].lower())
        img = cv.imread(img_path, 0)
        img = cv.resize(img, (imgW, imgH))
        img = np.transpose(img, (1, 0))
        img = torch.from_numpy(img / 255.)

        text, length = self.converter.encode(text)
        length = length[0]

        print('text.size(): ' + str(text.size()))
        print('length.size(): ' + str(length.size()))

        return img, text, length


if __name__ == "__main__":
    dataset = MJSynthDataset('val')
    print(dataset[1])
