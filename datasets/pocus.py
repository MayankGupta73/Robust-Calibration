import os
import random
import pickle
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

fold = 1
data_dir = "/covid_data{}.pkl".format(fold)

class COVIDDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        """
        POCUS Dataset
            param data_dir: str
            param transform: torch.transform
        """
        self.label_name = {"covid19": 0, "pneumonia": 1, "regular": 2}
        with open(data_dir, 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
        if train:
            self.X, self.y = X_train, y_train       # [N, C, H, W], [N]
        else:
            self.X, self.y = X_test, y_test         # [N, C, H, W], [N]
        self.transform = transform
    
    def __getitem__(self, index):
        img_arr = self.X[index].transpose(1,2,0)    # CHW => HWC
        img = Image.fromarray(img_arr.astype('uint8')).convert('RGB') # 0~255
        label = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.y)

                            
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
])

aug_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.25),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
])
                        
def get_train_valid_test_loader(args, tta = False):
    
    train_data = COVIDDataset(data_dir=data_dir, train=True, transform=train_transform)
    if tta:
        val_data = COVIDDataset(data_dir=data_dir, train=False, transform=aug_transform)
    else:
        val_data = COVIDDataset(data_dir=data_dir, train=False, transform=val_transform)

    train_loader = DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(dataset=val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=5)
    
    return train_loader, val_loader, val_loader
    
def get_datasets(args, tta = False):
    train_data = COVIDDataset(data_dir=data_dir, train=True, transform=train_transform)
    if tta:
        val_data = COVIDDataset(data_dir=data_dir, train=False, transform=aug_transform)
    else:
        val_data = COVIDDataset(data_dir=data_dir, train=False, transform=val_transform)
    
    #val_data = COVIDDataset(data_dir=data_dir, train=False, transform=val_transform)
    
    return train_data, val_data

def get_transforms():
    return train_transform, val_transform