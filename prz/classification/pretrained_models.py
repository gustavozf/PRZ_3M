import abc

import numpy as np
from tensorflow.keras import applications as keras_app

from prz.utils.image_sample import ImageSample

class CnnModel(abc.ABC):
    MODEL = None
    PREPROCESS_FUNC = None
    INPUT_SHAPE = None

    @classmethod
    def preprocess_sample(cls, data: np.array):
        return cls.PREPROCESS_FUNC(data)

    @classmethod
    def preprocess_data_array(
            cls,
            data: np.array,
        ):
        return np.array([
            cls.preprocess_sample(
                ImageSample.resize(img, cls.INPUT_SHAPE),
            )
            for img in data
        ])

    @classmethod
    def feature_extraction(
            cls,
            data: np.array,
            weights: str='imagenet',
        ):
        model = cls.MODEL(weights=weights, include_top=False)   
        processed_data = cls.preprocess_sample(data)

        return model.predic(processed_data)

class VGG16(CnnModel):
    MODEL = keras_app.VGG16
    PREPROCESS_FUNC = keras_app.vgg16.preprocess_input
    INPUT_SHAPE = None

class VGG19(CnnModel):
    MODEL = keras_app.VGG19
    PREPROCESS_FUNC = keras_app.vgg19.preprocess_input
    INPUT_SHAPE = None

class ResNet50V2(CnnModel):
    MODEL = keras_app.ResNet50V2
    PREPROCESS_FUNC = keras_app.resnet_v2.preprocess_input
    INPUT_SHAPE = None

class ResNet101V2(CnnModel):
    MODEL = keras_app.ResNet101V2
    PREPROCESS_FUNC = keras_app.resnet_v2.preprocess_input
    INPUT_SHAPE = None

class ResNet152V2(CnnModel):
    MODEL = keras_app.ResNet152V2
    PREPROCESS_FUNC = keras_app.resnet_v2.preprocess_input
    INPUT_SHAPE = None

class InceptionV3(CnnModel):
    MODEL = keras_app.InceptionV3
    PREPROCESS_FUNC = keras_app.inception_v3.preprocess_input
    INPUT_SHAPE = None

class InceptionResNetV2(CnnModel):
    MODEL = keras_app.InceptionResNetV2
    PREPROCESS_FUNC = keras_app.inception_resnet_v2.preprocess_input
    INPUT_SHAPE = None

class MobileNetV2(CnnModel):
    MODEL = keras_app.MobileNetV2
    PREPROCESS_FUNC = keras_app.mobilenet_v2.preprocess_input
    INPUT_SHAPE = None
