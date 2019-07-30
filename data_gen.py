import os
import random

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import sampler

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
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        print(img)

        text = str(img_path.split('_')[1].lower())
        print(text)

        return img, text


# class resizeNormalize(object):
#
#     def __init__(self, size, interpolation=Image.BILINEAR):
#         self.size = size
#         self.interpolation = interpolation
#         self.toTensor = transforms.ToTensor()
#
#     def __call__(self, img):
#         img = img.resize(self.size, self.interpolation)
#         img = self.toTensor(img)
#         img.sub_(0.5).div_(0.5)
#         return img
#
#
# class randomSequentialSampler(sampler.Sampler):
#
#     def __init__(self, data_source, batch_size):
#         print(type(data_source))
#         print(len(data_source))
#         self.num_samples = len(data_source)
#         self.batch_size = batch_size
#
#     def __iter__(self):
#         n_batch = len(self) // self.batch_size
#         tail = len(self) % self.batch_size
#         index = torch.LongTensor(len(self)).fill_(0)
#         for i in range(n_batch):
#             random_start = random.randint(0, len(self) - self.batch_size)
#             batch_index = random_start + torch.range(0, self.batch_size - 1)
#             index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
#         # deal with tail
#         if tail:
#             random_start = random.randint(0, len(self) - self.batch_size)
#             tail_index = random_start + torch.range(0, tail - 1)
#             index[(i + 1) * self.batch_size:] = tail_index
#
#         return iter(index)
#
#     def __len__(self):
#         return self.num_samples
#
#
# class alignCollate(object):
#
#     def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
#         self.imgH = imgH
#         self.imgW = imgW
#         self.keep_ratio = keep_ratio
#         self.min_ratio = min_ratio
#
#     def __call__(self, batch):
#         images, labels = zip(*batch)
#
#         imgH = self.imgH
#         imgW = self.imgW
#         if self.keep_ratio:
#             ratios = []
#             for image in images:
#                 w, h = image.size
#                 ratios.append(w / float(h))
#             ratios.sort()
#             max_ratio = ratios[-1]
#             imgW = int(np.floor(max_ratio * imgH))
#             imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
#
#         transform = resizeNormalize((imgW, imgH))
#         images = [transform(image) for image in images]
#         images = torch.cat([t.unsqueeze(0) for t in images], 0)
#
#         return images, labels


if __name__ == "__main__":
    dataset = MJSynthDataset('val')
    print(dataset[1])
