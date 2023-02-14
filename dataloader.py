from pycocotools.coco import COCO
from torch.utils.data import Dataset
from os.path import join, basename
import cv2
import numpy as np
import albumentations as A
import math
import torch
from albumentations.pytorch import ToTensorV2
import configs as cfg
import os

class GoalsDataset(Dataset):

    def __init__(self,annotation_path, images_path, s = 14, b = 2, number_classes = 4, image_size = 448, transforms = None):

        coco = COCO(annotation_path)
        cat_ids = coco.getCatIds(catNms='goal')
        images_ids = coco.getImgIds(catIds=cat_ids)
        annotations_ids = coco.getAnnIds(imgIds=images_ids)
        self.annotations = coco.loadAnns(annotations_ids)
        self.images = coco.loadImgs(images_ids)
        found_images = [x['id'] for x in self.images if os.path.exists(images_path+x['file_name'])]
        self.images = [x for x in self.images if x['id'] in found_images]
        self.annotations = [x for x in self.annotations if x['image_id'] in found_images]
        print(len(self.images), len(self.annotations))
        self.images_path = images_path

        self.transforms = transforms

        self.s = s
        self.b = b
        self.number_classes = number_classes
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = join(self.images_path,basename(self.images[idx]['path']))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        keypoints = np.array(self.annotations[idx]['keypoints']).reshape([-1,3])[:,:2]
        labels = list(range(self.number_classes))

        sample = {'image':image,'keypoints':keypoints, 'category_id':labels}

        if self.transforms is not None:
            sample = self.transforms(**sample)

        keypoints = sample['keypoints']
        image = sample['image']
        labels = sample['category_id']



        out = self.get_label(keypoints, labels)

        return image, out


    def get_label(self, keypoints, labels):

        # reads the labels and makes one label example
        out = np.zeros((self.s, self.s, self.number_classes + self.b * 3), dtype=np.float32)

        for i in range(len(keypoints)):
            idx = labels[i]

            x, y = int(keypoints[i][0]), int(keypoints[i][1])

            cell_size = self.image_size // self.s

            # TRIAL
            cell_x = x // cell_size
            cell_y = y // cell_size

            x %= cell_size
            y %= cell_size

            x /= cell_size
            y /= cell_size

            for i in range(self.b):
                out[cell_y, cell_x, 0 + 3 * i] = x
                out[cell_y, cell_x, 1 + 3 * i] = y
                out[cell_y, cell_x, 2 + 3 * i] = 1

            out[cell_y, cell_x, idx + self.b * 3] = 1

        # return one label of shape [grid size, gride size, classes+3]
        return out


def get_aug(aug):
    return A.Compose(aug, keypoint_params=A.KeypointParams(format='xy',label_fields=['category_id']) )


def get_train_test_data_loaders(dataset_path, batch_size, shuffle = True, num_workers = 8):

    train_trans = get_aug([
        A.Resize(448,448),
        # A.HorizontalFlip(p=.5),
        #A.RandomResizedCrop(width=224, height=224, interpolation=cv2.INTER_CUBIC),
        A.Rotate(30),
        A.RGBShift(p=.5),
        A.Blur(blur_limit=5, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])
    test_trans = get_aug([
        A.Resize(448, 448),
        A.Normalize(),
        ToTensorV2()
    ])
    train_dataset = GoalsDataset(dataset_path+'goals_pose_749.json', dataset_path+'train/',s = cfg.GRID_SIZE,b = cfg.BOXES_PER_CELL, transforms=train_trans)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=shuffle, num_workers=num_workers)

    test_dataloader = None
    if os.path.exists(dataset_path+'test'):

        test_dataset = GoalsDataset(dataset_path + 'goals_pose_749.json', dataset_path + 'test/', s=cfg.GRID_SIZE,
                                     b=cfg.BOXES_PER_CELL, transforms=test_trans)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)


    return train_dataloader, test_dataloader


if __name__ == "__main__":

    trans = get_aug([
        A.Resize(448, 448),
        # A.HorizontalFlip(p=1),
        # A.RandomResizedCrop(width=448, height=448, interpolation=cv2.INTER_CUBIC),
        A.Rotate(30, border_mode=cv2.BORDER_REPLICATE),
        A.RGBShift(p=.5),
        A.Blur(blur_limit=3, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(p=0.5),
        # A.Normalize(),
        # ToTensorV2()
    ])
    dataset = GoalsDataset('goals749/goals_pose_749.json', 'goals749/train/', transforms=trans)

    for i in range(100):
        img,k, l = dataset[i]
        print(k)
        print(l)
        cv2.imshow('img',img)
        cv2.waitKeyEx(-1)
