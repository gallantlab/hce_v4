"""
Test functionality of HCE model with example weight file.
Useful in docker containers to validate them.
"""

import time

start_time = time.time()
print("time started")
from kfs.generators import time_delay_generator
from hce_architecture_keras import create_model

import numpy as np

delays = range(2, 16)
np.random.seed(42)  # for reproducibility
batch_size = 400

weight_file = "example_hce_weight_file_keras.w"

print("weight file defined")

model = create_model()
print("model created")
model.load_weights(weight_file)
print("Model weights successfully loaded!")


X_test = np.random.rand(2000, 3, 64, 64)
print("made xtest")
val_gen = time_delay_generator(X_test, None, delays, batch_size, shuffle=False)
print("made valgen")
pred1 = model.predict_generator(val_gen, np.ceil(X_test.shape[0] / float(batch_size)))

print("Ran successful model prediction!")
executionTime = time.time() - start_time
print("Execution time in seconds: " + str(executionTime))
