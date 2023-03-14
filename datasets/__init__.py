from .cifar10 import get_train_valid_test_loader as cifar10loader
from .cifar10 import get_datasets as cifar10datasets
from .cifar10 import get_transforms as cifar10transforms

from .cifar100 import get_train_valid_test_loader as cifar100loader
from .cifar100 import get_datasets as cifar100datasets
from .cifar100 import get_transforms as cifar100transforms

from .imagenet import get_train_valid_test_loader as imagenetloader
from .imagenet import get_datasets as imagenetdatasets
from .imagenet import get_transforms as imagenettransforms

from .gbc_usg import get_train_valid_test_loader as gbcusgloader
from .gbc_usg import get_datasets as gbcusgdatasets
from .gbc_usg import get_transforms as gbcusgtransforms

from .db_ret import get_train_valid_test_loader as dbretloader
from .db_ret import get_datasets as dbretdatasets
from .db_ret import get_transforms as dbrettransforms

from .busi import get_train_valid_test_loader as busiloader
from .busi import get_datasets as busidatasets
from .busi import get_transforms as busitransforms

from .pocus import get_train_valid_test_loader as pocusloader
from .pocus import get_datasets as pocusdatasets
from .pocus import get_transforms as pocustransforms

from .covid_ct import get_train_valid_test_loader as covidctloader
from .covid_ct import get_datasets as covidctdatasets
from .covid_ct import get_transforms as covidcttransforms

from .melanoma import get_train_valid_test_loader as melanomaloader
from .melanoma import get_datasets as melanomadatasets
from .melanoma import get_transforms as melanomatransforms

dataloader_dict = {
    "cifar10" : cifar10loader,
    "cifar100" : cifar100loader,
    "imagenet" : imagenetloader,
    "gbc_usg" : gbcusgloader,
    "db_ret" : dbretloader,
    "busi": busiloader,
    "pocus": pocusloader,
    "covid_ct": covidctloader,
    "melanoma": melanomaloader,
}

dataset_dict = {
    "cifar10" : cifar10datasets,
    "cifar100" : cifar100datasets,
    "imagenet" : imagenetdatasets,
    "gbc_usg" : gbcusgdatasets,
    "db_ret" : dbretdatasets,
    "busi": busidatasets,
    "pocus": pocusdatasets,
    "covid_ct": covidctdatasets,
    "melanoma": melanomadatasets,
}

dataset_nclasses_dict = {
    "cifar10" : 10,
    "cifar100" : 100,
    "imagenet" : 200,
    "gbc_usg": 3,
    "db_ret" : 2,
    "busi": 3,
    "pocus": 3,
    "covid_ct" : 2,
    "melanoma" : 2,
}

dataset_classname_dict = {
    "cifar10" : ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],

    "cifar100" : ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud',
                'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 
                'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 
                'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
                'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                'whale', 'willow_tree', 'wolf', 'woman', 'worm'],

    "imagenet" : [f"{i}" for i in range(200)],
    
    "gbc_usg" : ['Nml', 'Ben', 'Mal'],
    
    "db_ret" : ['Nml', 'Mal'],
    
    "busi" : ['Nml', 'Ben', 'Mal'],
    
    "pocus" : ['Nml', 'Ben', 'Mal'],
    
    "covid_ct" : ['Covid', 'NonCovid'],

    "melanoma" : ['Ben', 'Mal'],
}

dataset_transform_dict = {
    "cifar10" : cifar10transforms,
    "cifar100" : cifar100transforms,
    "imagenet" : imagenettransforms,
    "gbc_usg" : gbcusgtransforms,
    "db_ret" : dbrettransforms,
    "busi": busitransforms,
    "pocus": pocustransforms,
    "covid_ct": covidcttransforms,
    "melanoma": melanomatransforms,
}
