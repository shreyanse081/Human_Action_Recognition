import numpy as np
from action_classification_models import C3D
from Classes_array_common import classes

if __name__ == '__main__':
    model = C3D()
    model.load_weights('C:\\tensorflow1\\models\\research\\object_detection\\action_classification_saved\\model175.hdf5')
    v = np.load('rgb.npy')[:, :, :, ::-1]
    p = [v for i in range(1)]
    p = np.reshape(p, newshape=(1, 20, 112, 112, 3))
    y = model.predict(p)
    y = np.sum(y, axis=0)
    print(classes[np.argmax(y / 1)])