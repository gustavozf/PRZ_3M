from abc import ABC, abstractmethod
import numpy as np

class Dataset(ABC):
    X_data = np.array([])
    y_label = np.array([])

    @abstractmethod
    def get_data(self):
        return NotImplementedError

    @abstractmethod
    def get_test_train(self):
        return NotImplementedError

    @abstractmethod
    def get_k_fold_cv(self):
        return NotImplementedError