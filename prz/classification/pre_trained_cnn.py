import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import applications as keras_app

from prz.definitions.configs import NeuralNetConfigs

class PreTrainedCnnModels:
    __models = {
        'VGG16': keras_app.VGG16,
        'VGG19': keras_app.VGG19,
        'ResNet50V2': keras_app.ResNet50V2,
        'ResNet101V2': keras_app.ResNet101V2,
        'ResNet152V2': keras_app.ResNet152V2,
        'InceptionV3': keras_app.InceptionV3,
        'InceptionResNetV2': keras_app.InceptionResNetV2,
        'MobileNetV2': keras_app.MobileNetV2
    }

    __preprocess_methods = {
        'VGG16': keras_app.vgg16.preprocess_input,
        'VGG19': keras_app.vgg19.preprocess_input,
        'ResNet50V2': keras_app.resnet_v2.preprocess_input,
        'ResNet101V2': keras_app.resnet_v2.preprocess_input,
        'ResNet152V2': keras_app.resnet_v2.preprocess_input,
        'InceptionV3': keras_app.inception_v3.preprocess_input,
        'InceptionResNetV2': keras_app.inception_resnet_v2.preprocess_input,
        'MobileNetV2': keras_app.mobilenet_v2.preprocess_input
    }

    @classmethod
    def __check_model_availability(cls, model_name: str):
        available_models = set(cls.__models.keys())
        assert model_name in available_models, (
            f'Model {model_name} not available among the supported models. '
            f'Currently supported models: {available_models}'
        )

    @classmethod
    def preprocess_data(cls, data: np.array, model_name: str='ResNet50V2'):
        cls.__check_model_availability(model_name)
        return cls.__preprocess_methods[model_name](data)

    @classmethod
    def feature_extraction(
            cls,
            data: np.array,
            model_name: str='ResNet50V2',
            weights: str='imagenet',
        ):
        cls.__check_model_availability(model_name)
        processed_data = PreTrainedCnnModels.preprocess_data(
            data, model_name=model_name
        )

        model = cls.__models[model_name](
            weights=weights, include_top=False
        )   

        return model.predic(processed_data)

    @classmethod
    def fine_tuning(
            cls,
            X_train: np.array, y_train: np.array,
            X_valid: np.array, y_valid: np.array,
            model_name: str='ResNet50V2',
            weights: str='imagenet',
            out_layers_config: dict=NeuralNetConfigs.default_out_layers,
            **kwargs
        ):
        # Check the output layer configuration dictionary
        required_out_fields = set(NeuralNetConfigs.default_out_layers.keys())
        assert required_out_fields.issubset(set(out_layers_config.keys())), (
            'Inputted output layers configuration dictionary must '
            f'present the following items: {required_out_fields}'
        )
        # Get the training arguments from **kwargs
        optimizer = kwargs.get('optimizer', NeuralNetConfigs.default_opt)
        verbose = kwargs.get('verbose', 1)
        epochs = kwargs.get('epochs', 64)
        batch_size = kwargs.get('batch_size', 32)
        callbacks = kwargs.get('callbacks', [])
        loss = kwargs.get('loss', 'categorical_crossentropy')
        last_layer = kwargs.get('last_layer', 'softmax')

        # Check if "model_name" is valid
        cls.__check_model_availability(model_name)
        # Process the input data
        X_train = np.array([
            PreTrainedCnnModels.preprocess_data(img, model_name=model_name)
            for img in X_train
        ])
        X_valid = np.array([
            PreTrainedCnnModels.preprocess_data(img, model_name=model_name)
            for img in X_train
        ])

        # Create the base pre-trained model
        base_model = cls.__models[model_name](
            weights=weights, include_top=False
        )

        # Create the output (dense) layers
        x = base_model.output
        for layer in range(out_layers_config['n_layers']):
            x = Dense(
                out_layers_config['num_units'][layer],
                activation=out_layers_config['activation']
            )(x)

        predictions = Dense(
            out_layers_config['n_classes'],
            activation=last_layer
        )(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze the input layers
        for layer in base_model.layers:
            layer.trainable = False

        # Train with the frozen layers
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
            callbacks=callbacks
        )

        # Unfreeze all of the model's layers
        for layer in model.layers:
            layer.trainable = True

        # Train with the whole network
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
            callbacks=callbacks
        )

        return model