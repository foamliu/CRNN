import os

import cv2 as cv
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from config import IMG_FOLDER, annotation_files, imgH, imgW


class MJSynthDataset(Dataset):

    def __init__(self, split):
        annotation_file = annotation_files[split]

        with open(annotation_file, 'r') as file:
            self.lines = file.readlines()

        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return self.lines

    def __getitem__(self, i):
        line = self.lines[i]
        img_path = line.split(' ')[0]
        img_path = os.path.join(IMG_FOLDER, img_path)
        text = str(img_path.split('_')[1].lower())
        img = cv.imread(img_path, 0)
        img = cv.resize(img, (imgW, imgH), cv.INTER_CUBIC)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)

        return img, text


if __name__ == "__main__":
    dataset = MJSynthDataset('val')
    print(dataset[1])
