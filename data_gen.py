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
    # print('delta_w: ' + str(delta_w))
    delta_h = height - new_size[0]
    # print('delta_h: ' + str(delta_h))
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_REPLICATE)
    # return the resized image
    return new_im


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
        idx = i
        while True:
            try:
                return self.get_data_record(idx)
            except cv.error:
                import random
                idx = random.randint(0, len(self.lines) - 1)

    def get_data_record(self, i):
        line = self.lines[i]
        img_path = line.split(' ')[0]
        img_path = os.path.join(IMG_FOLDER, img_path)
        img = cv.imread(img_path)
        img = image_resize(img, width=imgW, height=imgH, inter=cv.INTER_CUBIC)
        # img = cv.resize(img, (imgW, imgH), cv.INTER_CUBIC)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)

        text = str(img_path.split('_')[1])

        return img, text


if __name__ == "__main__":
    import json
    from tqdm import tqdm

    annotation_file = annotation_files['train']
    with open(annotation_file, 'r') as file:
        lines = file.readlines()

    lengths = []
    alphabet = set()

    for line in tqdm(lines):
        img_path = line.split(' ')[0]
        label = str(img_path.split('_')[1])
        lengths.append(len(label))
        alphabet = alphabet | set(label)

    insights = dict()
    insights['alphabet'] = list(alphabet)
    insights['lengths'] = lengths
    with open('insights.json', 'w') as file:
        json.dump(insights, file)

    print('len(alphabet): ' + str(len(alphabet)))
    print('max(lengths): ' + str(max(lengths)))
