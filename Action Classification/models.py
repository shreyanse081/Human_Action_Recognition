from tensorflow.keras import layers, Model
from tensorflow.keras.utils import multi_gpu_model


classes = 60


def get_model(x):
    x = layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
                      activation='relu')(x)
    x = layers.MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)
#    x = layers.SpatialDropout3D(0.35)(x)
    x = layers.Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                      activation='relu')(x)
    x = layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
#    x = layers.SpatialDropout3D(0.35)(x)
    x = layers.Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                      activation='relu')(x)
    x = layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
#    x = layers.SpatialDropout3D(0.35)(x)
    x = layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                      activation='relu')(x)
    x = layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2))(x)
#    x = layers.SpatialDropout3D(0.35)(x)
#    x = layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
#                      activation='relu')(x)
  #  x = layers.SpatialDropout3D(0.5)(x)
#    x = layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = layers.Flatten()(x)
    return x


def C3D():
    inp = layers.Input((20, 112, 112, 3))
    z = get_model(inp)
    z = layers.Dense(1024, activation='relu', name='fc1')(z)
    z = layers.Dropout(0.35)(z)
    z = layers.Dense(classes, activation='softmax', )(z)

    model = Model(inputs=inp, outputs=z)
    return model

