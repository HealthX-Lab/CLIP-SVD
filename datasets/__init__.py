from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .imagenetv2 import ImageNetV2
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch

from .btmri import BTMRI
from .busi import BUSI
from .ctkidney import CTKidney
from .chmnist import CHMNIST
from .kneexray import KneeXray
from .kvasir import Kvasir
from .lungcolon import LungColon
from .retina import RETINA
from .covid import COVID_19
from .dermamnist import DermaMNIST
from .octmnist import OCTMNIST


dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc_aircraft": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet": ImageNet,
                "imagenet_a": ImageNetA,
                "imagenetv2": ImageNetV2,
                "imagenet_r": ImageNetR,
                "imagenet_sketch": ImageNetSketch,
                "busi": BUSI,
                "btmri": BTMRI,
                "ctkidney": CTKidney,
                "chmnist": CHMNIST,
                "kneexray": KneeXray,
                "kvasir": Kvasir,
                "lungcolon": LungColon,
                "retina": RETINA,
                "covid": COVID_19,
                "dermamnist": DermaMNIST,
                "octmnist": OCTMNIST 
                }


def build_dataset(dataset, root_path, shots, template, subsample):
    return dataset_list[dataset](root_path, shots, template, subsample)