import numpy as np

from prz.classification.pre_trained_cnn import PreTrainedCnnModels

def extract_cnn_features(
        input_sample: np.array,
        model_name: str='ResNet50V2',
        weights: str='imagenet'
    ):
    features = PreTrainedCnnModels.feature_extraction(
        input_sample, model_name=model_name, weights=weights
    )

    return np.reshape(features, (np.prod(np.shape(features))))
