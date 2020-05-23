import json
import tensorflow as tf
import numpy as np
import os
import string

from flask import Flask, request

app = Flask(__name__)

model = tf.keras.models.load_model('app_nnv.h5')

# Keras Functional API to get output of each layer
app_model = tf.keras.models.Model(
    model.inputs,
    [layer.output for layer in model.layers]
)

_, (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.


def get_prediction():
    idx = np.random.choice(x_test.shape[0])
    image = x_test[idx, :, :]
    img_arr = np.reshape(image, (1, 784))

    return app_model.predict(img_arr), image


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        preds, img = get_prediction()
        # important to return lists for json
        final_preds = [p.tolist() for p in preds]
        return json.dumps({'prediction': final_preds, 'image': img.tolist()})
    return 'Welcome to the NNVisualiser server!'


if __name__ == '__main__':
    app.run()
