import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input 
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input 

class CNNModel():

    def __init__(self, 
        base_model=None,
        input_shape = (),
        min_input_shape = (),
        preprocess_input = None,
    ): 
        self.base_model = base_model
        self.input_shape = input_shape
        self.min_input_shape = min_input_shape
        self.preprocess_input = preprocess_input

    def predict(self, img=np.array([])):
        return self.base_model.predict(img)


class PreTrainedCNNModels():
    @staticmethod
    def VGG16(weights='imagenet', input_shape=(224, 224, 3)):
        min_input_shape=(32, 32, 3)
        
        if (input_shape == 'min'):
            input_shape = min_input_shape

        base_model = VGG16(weights=weights, 
                           include_top=False, 
                           input_shape=input_shape)

        return CNNModel(
            base_model=base_model,           
            input_shape=input_shape,           
            min_input_shape=min_input_shape,
            preprocess_input=preprocess_input,           
        )

    @staticmethod
    def InceptionV3(weights='imagenet', input_shape=(299, 299, 3)):
        min_input_shape=(75, 75, 3)

        if (input_shape == 'min'):
            input_shape = min_input_shape

        base_model = InceptionV3(weights=weights, 
                                 include_top=False,
                                 input_shape=input_shape)

        return CNNModel(
            base_model=base_model,           
            input_shape=input_shape,           
            min_input_shape=min_input_shape,
            preprocess_input=preprocess_input,           
        )

    @staticmethod
    def InceptionResNetV2(weights='imagenet', input_shape=(299, 299, 3)):
        min_input_shape=(75, 75, 3)

        if (input_shape == 'min'):
            input_shape = min_input_shape

        base_model = InceptionResNetV2(weights=weights, 
                                       include_top=False,
                                       input_shape=input_shape)

        return CNNModel(
            base_model=base_model,           
            input_shape=input_shape,           
            min_input_shape=min_input_shape,
            preprocess_input=preprocess_input,           
        )

class CNNPredictor():
    def __init__(self, model=PreTrainedCNNModels.VGG16()): 
        self.model = model

    def summary(self):
        return self.model.base_model.summary()

    def extract_features(self, imgs=np.array([])):
        features = self.model.predict(imgs)
        return np.reshape(features, (np.prod(np.shape(features))))
