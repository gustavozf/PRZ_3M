from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam

class NeuralNetConfigs:
    default_opt = Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        decay=0.0005
    )
    default_out_layers = {
        'n_layers': 4,
        'num_units': [1024, 512, 256, 128],
        'activation': LeakyReLU(alpha=0.01),
        'n_classes': 2,
        'last_layer_actv': 'softmax',
    }
