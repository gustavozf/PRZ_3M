import numpy as np

class PreTrainedCNNModels():
    @staticmethod
    def VGG16(weights='imagenet', input_shape=(224, 224, 3)):
        from tensorflow.keras.applications.vgg16 import VGG16
        base_model = VGG16(weights=weights, 
                           include_top=False, 
                           input_shape=input_shape)

        return base_model, input_shape

    @staticmethod
    def InceptionV3(weights='imagenet', input_shape=(299, 299, 3)):
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(weights=weights, 
                                 include_top=False,
                                 input_shape=input_shape)

        return base_model, input_shape

    @staticmethod
    def InceptionResNetV2(weights='imagenet', input_shape=(299, 299, 3)):
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
        base_model = InceptionResNetV2(weights=weights, 
                                       include_top=False,
                                       input_shape=input_shape)

        return base_model, input_shape

class CNNPredictor():
    def __init__(self, model=PreTrainedCNNModels.VGG16()): 
        self.model, self.input_shape = model

    def extract_features(self, imgs=np.array([])):
        features = self.model.predict(imgs)
        return np.reshape(features, (np.prod(np.shape(features))))
