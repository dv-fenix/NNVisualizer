import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize Data
x_train = np.reshape(x_train, (60000, 28 * 28))
x_train = x_train / 255.

x_test = np.reshape(x_test, (10000, 28 * 28))
x_test = x_test / 255.

# Define model using Keras Sequential API
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(784,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# History Object not required
_ = model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=30, batch_size=128,
              verbose=2)

# Save model weigths in the same directory as app.py and server.py
model.save('app_nnv.h5')
