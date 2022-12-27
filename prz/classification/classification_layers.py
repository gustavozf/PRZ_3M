from tensorflow.keras.layers import (
    Dense, BatchNormalization, Dropout, LeakyReLU
)

def lenet5(x, n_classes):
    x = Dense(120, activation = 'relu')(x)
    x = Dense(84, activation = 'relu')(x)
    
    return Dense(n_classes, activation = 'softmax')(x)

def alexnet(x, n_classes):
    x = Dense(4096, activation = 'relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)

    # 2nd Dense Layer
    x = Dense(4096, activation = 'relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)

    # 3rd Dense Layer
    x = Dense(1000, activation = 'relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    
    return Dense(n_classes, activation = 'softmax')(x)

def roeckernet(x, n_classes):
    x = Dense(4096, activation=LeakyReLU(alpha=0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.01))(x)
    
    return Dense(n_classes, activation='softmax')(x)