import os
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset

from config import IMG_FOLDER, annotation_files, imgH, imgW


class MJSynthDataset(Dataset):

    def __init__(self, split):
        annotation_file = annotation_files[split]

        print('loading {} annotation data...'.format(split))
        with open(annotation_file, 'r') as file:
            self.lines = file.readlines()

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
        img = cv.imread(img_path, 0)
        img = cv.resize(img, (imgW, imgH), cv.INTER_CUBIC)
        img = np.transpose(img, (1, 0))
        img = np.reshape(img, (1, imgH, imgW))

        img = img[..., ::-1]  # RGB
        img = img / 255. - 0.5

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
