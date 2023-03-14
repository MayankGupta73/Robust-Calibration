import cv2
import os
import torch
import json
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import datasets.transforms as T
from torchvision.transforms import functional as F
import torchvision.transforms as tfm
from PIL import Image

meta_file = "path to json file"
set_dir = "dataset path"
train_set_name = "train split txt name"
test_set_name = "val split txt name"
img_dir = "dataset images path"
batch_size = 16
width = 224
height = 224

class BUSIRawDataset(Dataset):
    """ GB classification dataset. """
    def __init__(self, img_dir, df, labels, img_transforms=None):
        self.img_dir = img_dir
        self.transforms = img_transforms
        d = []
        for label in labels:
            key, cls = label.split(",")
            val = df[key]
            val["filename"] = key
            val["label"] = int(cls)
            d.append(val)
        self.df = d

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image
        filename = self.df[idx]["filename"]
        img_name = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_name)
        if self.transforms:
            img = self.transforms(image)
        label = torch.as_tensor(self.df[idx]["label"], dtype=torch.int64)
        #cv2.imwrite(filename, image)
        print
        return img, label
        
class BUSIDataset(Dataset):
    """ GB classification dataset. """
    def __init__(self, img_dir, df, labels, is_train=True, to_blur=True, blur_kernel_size=(65,65), sigma=0, p=0.15, img_transforms=None):
        self.img_dir = img_dir
        self.transforms = img_transforms
        self.to_blur = to_blur
        self.blur_kernel_size = blur_kernel_size
        self.sigma = sigma
        self.is_train = is_train
        d = []
        for label in labels:
            key, cls = label.split(",")
            val = df[key]
            val["filename"] = key
            val["label"] = int(cls)
            d.append(val)
        self.df = d
        self.p = p

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image
        filename = self.df[idx]["filename"]
        img_name = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_name)
        if self.to_blur:
            image = cv2.GaussianBlur(image, self.blur_kernel_size, self.sigma)
        image = crop_image(image, self.df[idx]["Gold"], self.p)
        if self.transforms:
            image = self.transforms(image)
        label = torch.as_tensor(self.df[idx]["label"], dtype=torch.int64)
        return image, label

class BUSICropDataset(Dataset):
    """ GB classification dataset. """
    def __init__(self, img_dir, df, labels, to_blur=True, blur_kernel_size=(65,65), sigma=16, p=0.15, img_transforms=None):
        self.img_dir = img_dir
        self.transforms = img_transforms
        self.to_blur = to_blur
        self.blur_kernel_size = (4*sigma+1, 4*sigma+1)#blur_kernel_size
        self.sigma = sigma
        self.p = p
        d = []
        for label in labels:
            key, cls = label.split(",")
            val = df[key]
            val["filename"] = key
            val["label"] = int(cls)
            d.append(val)
        self.df = d

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image
        filename = self.df[idx]["filename"]
        img_name = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_name)
        if self.to_blur:
            image = cv2.GaussianBlur(image, self.blur_kernel_size, self.sigma)
        orig = crop_image(image, self.df[idx]["Gold"], self.p)
        if self.transforms:
            orig = self.transforms(orig)
        # Get the roi bbox
        num_objs = len(self.df[idx]["Boxes"])
        label = torch.as_tensor(self.df[idx]["label"], dtype=torch.int64)
        crps = []
        labels = []
        for i in range(num_objs):
            bbs = self.df[idx]["Boxes"][i]
            crp_img = crop_image(image, bbs, self.p)
            #stack the predicted rois as different samples
            if self.transforms:
                crp_img = self.transforms(crp_img)
            crps.append(crp_img)
            labels.append(label)
        if num_objs == 0:
            #use the original img if no bbox predicted
            #orig = self.transforms(image)
            orig = orig.unsqueeze(0)
            label = label.unsqueeze(0)
        else:
            orig = torch.stack(crps, 0)
            label = torch.stack(labels, 0)
        return orig, label
        
def crop_image(image, box, p):
    x1, y1, x2, y2 = box
    cropped_image = image[int((1-p)*y1):int((1+p)*y2), \
                            int((1-p)*x1):int((1+p)*x2)]
    return cropped_image


transforms = []
transforms.append(T.Resize((width, height)))
#transforms.append(T.RandomHorizontalFlip(0.25))
transforms.append(T.ToTensor())
img_transforms = T.Compose(transforms)

val_transforms = T.Compose([T.Resize((width, height)),\
                            T.ToTensor()])
                            
aug_transforms = tfm.Compose([tfm.ToPILImage(),
                            tfm.Resize((width, height)),
                            tfm.RandomHorizontalFlip(0.25),
                            tfm.RandomRotation(10),
                            tfm.ToTensor()])
                            
def get_train_valid_test_loader(args, tta = False, to_blur=False, blur_kernel_size=(65,65), sigma=16):
    with open(meta_file, "r") as f:
        df = json.load(f)

    train_labels = []
    t_fname = os.path.join(set_dir, train_set_name)
    with open(t_fname, "r") as f:
        for line in f.readlines():
            train_labels.append(line.strip())
    val_labels = []
    v_fname = os.path.join(set_dir, test_set_name)
    with open(v_fname, "r") as f:
        for line in f.readlines():
            val_labels.append(line.strip())
            
    if tta:
        val_t = aug_transforms
        val_dataset = BUSICropDataset(img_dir, df, val_labels, to_blur=False, img_transforms=val_t)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=5)
    else:
        val_t = val_transforms
        val_dataset = BUSICropDataset(img_dir, df, val_labels, to_blur=False, img_transforms=val_t)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=5)
    
    train_dataset = BUSIDataset(img_dir, df, train_labels, to_blur=False, img_transforms=img_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=4)
    
    #val_dataset = BUSICropDataset(img_dir, df, val_labels, to_blur=False, img_transforms=val_transforms)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=5)
    
    return train_loader, val_loader, val_loader
    
def get_datasets(args, tta = False, to_blur=False, blur_kernel_size=(65,65), sigma=16):
    with open(meta_file, "r") as f:
        df = json.load(f)

    train_labels = []
    t_fname = os.path.join(set_dir, train_set_name)
    with open(t_fname, "r") as f:
        for line in f.readlines():
            train_labels.append(line.strip())
    val_labels = []
    v_fname = os.path.join(set_dir, test_set_name)
    with open(v_fname, "r") as f:
        for line in f.readlines():
            val_labels.append(line.strip())
    
    train_dataset = BUSIDataset(img_dir, df, train_labels, img_transforms=img_transforms)
    
    if tta:
        val_t = aug_transforms
        val_dataset = BUSICropDataset(img_dir, df, val_labels, to_blur=False, img_transforms=val_t)
    else:
        val_t = val_transforms
        val_dataset = BUSICropDataset(img_dir, df, val_labels, to_blur=False, img_transforms=val_t)
    #val_dataset = BUSIDataset(img_dir, df, val_labels, img_transforms=val_transforms)
    
    return train_dataset, val_dataset

def get_transforms():
    return img_transforms, val_transforms