import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==========================================================
# Training Dataset
# ==========================================================

class CamObjDataset(data.Dataset):

    def __init__(self, image_root, gt_root, trainsize):

        self.trainsize = trainsize

        self.images = [
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        self.gts = [
            os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.filter_files()

        self.size = len(self.images)

        self.transform = A.Compose([

            A.Resize(trainsize, trainsize),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.3),

            A.Rotate(
                limit=15,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5
            ),

            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT,
                p=0.4
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),

            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),

            A.GaussianBlur(
                blur_limit=(3,5),
                p=0.2
            ),

            A.GaussNoise(
                std_range=(0.02,0.08),
                p=0.2
            ),

            A.Normalize(
                mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)
            ),

            ToTensorV2()

        ])

    def __getitem__(self, index):

        image = np.array(self.rgb_loader(self.images[index]))
        gt = np.array(self.binary_loader(self.gts[index]))

        augmented = self.transform(
            image=image,
            mask=gt
        )

        image = augmented["image"]
        gt = augmented["mask"].unsqueeze(0).float() / 255.0

        return image, gt

    def filter_files(self):

        images = []
        gts = []

        for img_path, gt_path in zip(self.images, self.gts):

            img = Image.open(img_path)
            gt = Image.open(gt_path)

            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)

        self.images = images
        self.gts = gts

    def rgb_loader(self, path):

        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):

        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert("L")

    def __len__(self):

        return self.size


# ==========================================================
# Test Dataset
# ==========================================================

class test_dataset:

    def __init__(self, image_root, gt_root, testsize):

        self.testsize = testsize

        self.images = sorted([
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.lower().endswith((".jpg",".jpeg",".png"))
        ])

        self.gts = sorted([
            os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if f.lower().endswith((".jpg",".jpeg",".png"))
        ])

        self.transform = transforms.Compose([
            transforms.Resize((testsize,testsize)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

        self.size = min(len(self.images), len(self.gts))
        self.index = 0

    def load_data(self):

        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = os.path.basename(self.images[self.index])
        name = os.path.splitext(name)[0] + ".png"

        self.index += 1

        return image, gt, name

    def rgb_loader(self, path):

        with open(path,'rb') as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):

        with open(path,'rb') as f:
            img = Image.open(f)
            return img.convert("L")


# ==========================================================
# Faster Test Loader
# ==========================================================

class test_loader_faster(data.Dataset):

    def __init__(self, image_root, testsize):

        self.images = sorted([
            os.path.join(image_root,f)
            for f in os.listdir(image_root)
            if f.lower().endswith((".jpg",".jpeg",".png"))
        ])

        self.transform = transforms.Compose([
            transforms.Resize((testsize,testsize)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

    def __getitem__(self,index):

        image = self.rgb_loader(self.images[index])
        image = self.transform(image)

        return image, self.images[index]

    def rgb_loader(self,path):

        with open(path,'rb') as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __len__(self):

        return len(self.images)


# ==========================================================
# DataLoader
# ==========================================================

def get_loader(
    image_root,
    gt_root,
    batchsize,
    trainsize,
    shuffle=True,
    num_workers=0,
    pin_memory=True
):

    dataset = CamObjDataset(
        image_root,
        gt_root,
        trainsize
    )

    data_loader = data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return data_loader