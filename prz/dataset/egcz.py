import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from tensorflow.keras.utils import to_categorical

from prz.utils.image_sample import ImageSample

img_reader = np.vectorize(lambda img : ImageSample.read(img))

class EgczDataset(object):
    def __init__(
            self,
            fpaths: np.array,
            data: np.array,
            label: np.array,
            groups: np.array):
        self.file_path = fpaths
        self.data = data
        self.groups = groups
        self.n_groups = len(set(groups))
        self.n_classes = len(set(label))
        self.label = to_categorical(label, num_classes=self.n_classes)

    @staticmethod
    def from_csv(fpath: str):
        df = pd.read_csv(fpath)

        return EgczDataset(
            df['file_path'].to_numpy(),
            df['file_path'].apply(ImageSample.read).to_numpy(),
            df['label'].to_numpy(),
            df['anim_tag_id'].to_numpy()
        )

    def group_kfold_cv(self, n_splits:int=6):
        group_kfold = GroupKFold(n_splits=n_splits)
        return group_kfold.split(self.data, self.label, self.groups)

    def leave_one_out_kfold_cv(self):
        return self.group_kfold_cv(n_splits=self.n_groups)