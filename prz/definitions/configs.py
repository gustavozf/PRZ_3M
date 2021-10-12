from tensorflow.keras.optimizers import Adam

class NeuralNetConfigs:
    default_opt = Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        decay=0.0005
    )
    default_out_layers = {
        'n_layers': 2,
        'num_units': [1024, 512],
        'activation': 'relu',
        'n_classes': 2,
    }
