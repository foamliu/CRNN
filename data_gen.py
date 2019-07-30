import os

import cv2 as cv
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from config import IMG_FOLDER, annotation_files, imgH, imgW

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class MJSynthDataset(Dataset):

    def __init__(self, split):
        annotation_file = annotation_files[split]

        print('loading {} annotation data...'.format(split))
        with open(annotation_file, 'r') as file:
            self.lines = file.readlines()

        self.transformer = data_transforms[split]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        line = self.lines[i]
        img_path = line.split(' ')[0]
        img_path = os.path.join(IMG_FOLDER, img_path)
        img = cv.imread(img_path)
        img = cv.resize(img, (imgW, imgH), cv.INTER_CUBIC)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)

        text = str(img_path.split('_')[1].lower())

        return img, text


if __name__ == "__main__":
    dataset = MJSynthDataset('val')
    print(dataset[1])
