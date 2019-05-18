from tensorflow.keras import layers, Model
from tensorflow.keras.utils import multi_gpu_model

#
# def mean(tensors):
#     return (tensors[0] + tensors[1]) * 0.5


# def _shape(input_shape):
#     shape = list(input_shape)[0]
#     return shape


classes = 60


def get_model(x):
    x = layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
                      activation='relu')(x)
    x = layers.MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    x = layers.Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                      activation='relu')(x)
    x = layers.SpatialDropout3D(0.4)(x)
    x = layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = layers.Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                      activation='relu')(x)
    x = layers.SpatialDropout3D(0.4)(x)
    x = layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                      activation='relu')(x)
    x = layers.SpatialDropout3D(0.5)(x)
    x = layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                      activation='relu')(x)
    x = layers.SpatialDropout3D(0.5)(x)
    x = layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = layers.Flatten()(x)
    return x


def C3D():
    inp1 = layers.Input((20, 112, 112, 3))
    #    inp2 = layers.Input((20, 224, 224, 3,))

    z = get_model(inp1)
    #    y = get_model(inp2)

    #    z = layers.Lambda(mean, _shape)([z, y])
    z = layers.Dense(2048, activation='relu', name='fc1')(z)
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(2048, activation='relu', name='fc2')(z)
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(classes, activation='softmax', )(z)

    model = Model(inputs=inp1, outputs=z)
    # model = multi_gpu_model(model, gpus=2)  # Multi GPU Support
    return model