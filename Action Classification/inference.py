from tensorflow.keras.models import load_model
import numpy as np

classes = [0 for i in range(60)]
classes[0] = 1
video = np.load('processed/back_pain/rgb0.npy')

model = load_model('save_regularized/model.hdf5')

y = model.predict(video)
print(y)
