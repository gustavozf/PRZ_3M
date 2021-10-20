import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from tensorflow.keras.utils import to_categorical

from prz.utils.image_sample import ImageSample

img_reader = np.vectorize(lambda img : ImageSample.read(img))

class EgczDataset(object):
    def __init__(self, data: list, label: list, groups: list):
        self.data = data
        self.label = label
        self.groups = groups
        self.num_animals = len(set(groups))

    @staticmethod
    def from_csv(fpath: str):
        df = pd.read_csv(fpath)

        return EgczDataset(
            df['file_path'].apply(ImageSample.read).to_numpy(),
            to_categorical(df['label'].to_numpy(), num_classes=2),
            df['anim_tag_id'].to_numpy()
        )

    def group_kfold_cv(self, n_splits:int=6):
        group_kfold = GroupKFold(n_splits=n_splits)
        return group_kfold.split(self.data, self.label, self.groups)

    def leave_one_out_kfold_cv(self):
        return self.group_kfold_cv(n_splits=self.num_animals)