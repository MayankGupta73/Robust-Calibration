import os
import torch
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

input_size = 224
data_dir = "dataset path"

train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
aug_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(0.25),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_train_valid_test_loader(args, tta = False):
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), val_transforms)
    #test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), val_transforms)
    
    if tta:
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), aug_transform)
    else:
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), val_transforms)
                
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, val_loader, test_loader

def get_datasets(args, tta = False):
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), val_transforms)
    #test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), val_transforms)
    
    if tta:
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), aug_transform)
    else:
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), val_transforms)
    
    return train_dataset, test_dataset

def get_transforms():
    return train_transforms, val_transforms