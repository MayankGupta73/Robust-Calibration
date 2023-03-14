import os
import torch
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

input_size = 224
bs = 256
data_dir = "dataset path (with modified splits)"

train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.ColorJitter(brightness=0.125, contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
aug_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.ColorJitter(brightness=0.125, contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_train_valid_test_loader(args, tta = False):
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transforms)
    if tta:
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), aug_transforms)
    else:
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), val_transforms)
                
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.workers)

    return train_loader, val_loader, val_loader

def get_datasets(args, tta = False):
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transforms)
    #val_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), val_transforms)
    if tta:
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), aug_transforms)
    else:
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), val_transforms)
    
    return train_dataset, val_dataset

def get_transforms():
    return train_transforms, val_transforms