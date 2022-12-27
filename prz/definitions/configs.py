from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam

class NeuralNetConfigs:
    DEFAULT_OPT = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        decay=0.0005
    )