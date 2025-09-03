import os
import pickle

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase
from .dtd import DescribableTextures as DTD

class Kvasir(DatasetBase):

    dataset_dir = "Kvasir"

    def __init__(self, root, num_shots, template, subsample):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'Kvasir')
        self.split_path = os.path.join(self.dataset_dir, 'split_Kvasir.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)