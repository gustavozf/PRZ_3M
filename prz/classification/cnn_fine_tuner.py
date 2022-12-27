import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

from prz.classification import pretrained_models
from prz.definitions.configs import NeuralNetConfigs
from prz.classification import classification_layers

class CnnFineTuner:
    MODELS = {
        'VGG16': pretrained_models.VGG16,
        'VGG19': pretrained_models.VGG19,
        'ResNet50V2': pretrained_models.ResNet50V2,
        'ResNet101V2': pretrained_models.ResNet101V2,
        'ResNet152V2': pretrained_models.ResNet152V2,
        'InceptionV3': pretrained_models.InceptionV3,
        'InceptionResNetV2': pretrained_models.InceptionResNetV2,
        'MobileNetV2': pretrained_models.MobileNetV2
    }
    CLF_LAYERS = {
        'alexnet': classification_layers.alexnet,
        'roeckernet': classification_layers.roeckernet,
        'lenet5': classification_layers.roeckernet,
    }

    @classmethod
    def __check_model_availability(cls, model_name: str):
        available_models = set(cls.MODELS.keys())
        assert model_name in available_models, (
            f'Model {model_name} not available among the supported models. '
            f'Currently supported models: {available_models}'
        )
    
    @classmethod
    def __create_models(
            cls,
            model_name: str,
            weights: str,
            clf_layers: str,
            n_classes:int=2,
        ):
        # Create the base pre-trained model
        base_model = cls.MODELS[model_name].MODEL(
            weights=weights, include_top=False
        )

        # Create the output (dense) layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = cls.CLF_LAYERS[clf_layers](x, n_classes)

        model = Model(inputs=base_model.input, outputs=predictions)

        return base_model, model

    @classmethod
    def fine_tuning(
            cls,
            X_train: np.array, y_train: np.array,
            X_valid: np.array, y_valid: np.array,
            model_name: str='ResNet50V2',
            clf_layers:str='MAXNET',
            weights: str='imagenet',
            n_classes: int=2,
            **kwargs
        ):
        # Check the output layer configuration dictionary
        available_clf_layers = set(cls.CLF_LAYERS.keys())
        assert clf_layers in available_clf_layers, (
            'Inputted output layers configuration must '
            f'present the following items: {available_clf_layers}'
        )

        # Get the training arguments from **kwargs
        optimizer = kwargs.get('optimizer', NeuralNetConfigs.DEFAULT_OPT)
        verbose = kwargs.get('verbose', 1)
        epochs = kwargs.get('epochs', 64)
        batch_size = kwargs.get('batch_size', 32)
        callbacks = kwargs.get('callbacks', [])
        loss = kwargs.get('loss', 'categorical_crossentropy')

        # Check if "model_name" is valid
        cls.__check_model_availability(model_name)
        base_model, model = cls.__create_models(
            model_name, weights, clf_layers, n_classes=n_classes
        )

        # Freeze the input layers
        for layer in base_model.layers:
            layer.trainable = False

        # Train with the frozen layers
        print('Training the classification layers...')
        model.compile(
            optimizer=optimizer,
            loss=loss,
        )
        model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Unfreeze all of the model's layers
        for layer in model.layers:
            layer.trainable = True

        # Train with the whole network
        print('Training the whole network...')
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        return history, model
