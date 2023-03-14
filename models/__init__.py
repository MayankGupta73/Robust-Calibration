from .resnet import resnet20, resnet32, resnet56, resnet110
from .resnet_old_mc import resnet56_mc
from .resnet_imagenet import resnet34, resnet50
from .gbcnet import gbcnet, resnet50_gbc
from .resnet_mc import resnet50_mc_dropout, resnet50_mc_dropout_2

model_dict = {
    # resnet models can be used for cifar10/100, svhn
    # mendley models only to be used for mendley datasets

    "resnet20" : resnet20,
    "resnet32" : resnet32,
    "resnet56" : resnet56,
    "resnet110" : resnet110,
    
    "resnet56_mc" : resnet56_mc,

    "resnet34_imagenet" : resnet34,
    "resnet50_imagenet" : resnet50,
    
    "gbcnet" : gbcnet,
    "resnet50_gbc" : resnet50_gbc,
    "resnet50_mc": resnet50_mc_dropout,
    "resnet50_mc2": resnet50_mc_dropout_2,
}