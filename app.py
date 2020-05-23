import requests
import json
import numpy as np
import streamlit as st
import os
import matplotlib.pyplot as plt

URL = 'http://127.0.0.1:5000/'  # Change to path of the development server on your local machine

st.title('Neural Network Visualizer')
st.sidebar.markdown('## Input Image ##')

if st.button('Get Predictions'):
    response = requests.post(URL, data={})
    response = json.loads(response.text)
    # load predictions and image from server
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28, 28))
    # input image displayed in sidebar
    st.sidebar.image(image, width=150)

    for ps_layer, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))

        plt.figure(figsize=(32, 4))
        layer = 0
        if ps_layer == 0:
            continue
        elif ps_layer == 1:
            row = 4
            col = 16
        elif ps_layer == 2:
            continue
        elif ps_layer == 3:
            row = 2
            col = 16
            layer += 1
        elif ps_layer == 4:
            row = 1
            col = 16
            layer += 1
        else:
            row = 1
            col = 10
            layer += 1

        for i, number in enumerate(numbers):
            plt.subplot(row, col, i + 1)
            plt.imshow((number * np.ones((8, 8, 3))).astype('float32'), cmap='binary')
            plt.xticks([])
            plt.yticks([])
            if ps_layer == 5:
                plt.xlabel(str(i), fontsize=40)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()

        st.text('Layer {}'.format(layer + 1), )
        st.pyplot()
