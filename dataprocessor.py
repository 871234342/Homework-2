import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import random
import torch


class DatawAnnotation(Dataset):
    def __init__(self, root_dir, anno_dir, transform=None):

        self.root_dir = root_dir
        self.bboxs = pd.read_csv(anno_dir)
        self.transform = transform

        self.calsses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.img_ids = np.unique(self.bboxs['img_name'].to_numpy())

    def __len__(self):

        return len(self.img_ids)

    def load_img(self, idx):

        path = os.path.join(self.root_dir, self.img_ids[idx])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotation(self, idx):

        # Get groud truth
        img_id = self.img_ids[idx]
        anno_input = self.bboxs[self.bboxs['img_name'] == img_id]

        # Put bboxs for one image together and return
        annotations = np.zeros((0, 5))

        for i in range(len(anno_input.index)):
            annotation = np.zeros((1, 5))
            row_data = anno_input.iloc[i, :]
            annotation[0, 0] = row_data['left']
            annotation[0, 1] = row_data['top']
            annotation[0, 2] = row_data['right']
            annotation[0, 3] = row_data['bottom']
            annotation[0, 4] = row_data['label'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def __getitem__(self, idx):

        img = self.load_img(idx)
        annot = self.load_annotation(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample


class NoiseAdder(object):

    def add_noise(self, img):
        L = random.random() / 2
        img = img * (1 - L) + L * np.random.randn(img.shape[0], img.shape[1], img.shape[2])
        return img

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image = self.add_noise(image)

        sample = {'img': image, 'annot': annots}

        return sample





'''dataset = DatawAnnotation('train/', 'annotation.csv')

sample = dataset[0]
img = sample['img']
annot = sample['annot']

print(annot)
fig, ax = plt.subplots(1)
ax.imshow(img)

for box in annot:
    print(box)
    rect = patches.Rectangle(
        (box[0], box[3]), box[2] - box[0], box[1] - box[3],
        linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()'''