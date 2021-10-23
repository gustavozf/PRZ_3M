import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from prz.classification import pretrained_models
from prz.definitions.configs import NeuralNetConfigs

class CnnFineTuner:
    __models = {
        'VGG16': pretrained_models.VGG16,
        'VGG19': pretrained_models.VGG19,
        'ResNet50V2': pretrained_models.ResNet50V2,
        'ResNet101V2': pretrained_models.ResNet101V2,
        'ResNet152V2': pretrained_models.ResNet152V2,
        'InceptionV3': pretrained_models.InceptionV3,
        'InceptionResNetV2': pretrained_models.InceptionResNetV2,
        'MobileNetV2': pretrained_models.MobileNetV2
    }

    @classmethod
    def __check_model_availability(cls, model_name: str):
        available_models = set(cls.__models.keys())
        assert model_name in available_models, (
            f'Model {model_name} not available among the supported models. '
            f'Currently supported models: {available_models}'
        )

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

        # Check if "model_name" is valid
        cls.__check_model_availability(model_name)
        base_model, model = cls.__create_models(
            model_name, weights, out_layers_config,
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
    
    @classmethod
    def __create_models(
            cls, model_name: str, weights: str, out_layers_config: dict,
        ):
        # Create the base pre-trained model
        base_model = cls.__models[model_name](
            weights=weights, include_top=False
        )

        # Create the output (dense) layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        for layer in range(out_layers_config['n_layers']):
            x = Dense(
                out_layers_config['num_units'][layer],
                activation=out_layers_config['activation']
            )(x)

        predictions = Dense(
            out_layers_config['n_classes'],
            activation=out_layers_config['last_layer_actv']
        )(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        return base_model, model
