from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import random

app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model('your_model.h5')


def related_images(tag):
    folder_path = os.path.join('tiles',tag)
    result = [os.path.join('tiles',tag,item) for item in os.listdir(folder_path)]
    return random.sample(result, 6)


def preprocess_image(image_path):
    img = Image.open(image_path)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    resize = tf.image.resize(img, (128, 128))
    img = resize / 255.0  # Remove np.expand_dims
    return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the POST request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'})

        image = request.files['image']

        # Preprocess the image
        image_data = preprocess_image(image)

        # Make predictions
        predictions = model.predict(np.expand_dims(image_data, axis=0))

        maxi = np.argmax(predictions)

        avail_labels = ['tiles', 'fans', 'light_bulb']
        # print(predictions)

        resp = related_images(avail_labels[maxi])
        # print(resp, 'sameer')
        # Return the predictions as JSON
        response = {'data': resp}
        print(jsonify(response))
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/test_predict', methods=['GET'])
def test_predict():
    try:
        # Use an image from the "images" folder for testing
        test_image_path = 'images/image.jpeg'  # Adjust the path to the image you want to test

        # Preprocess the test image
        test_image_data = preprocess_image(test_image_path)

        # Make predictions
        predictions = model.predict(np.expand_dims(test_image_data, axis=0))

        # Return the predictions as JSON
        response = {'predictions': predictions.tolist()}
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(jsonify(response))
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/', methods=['GET'])
def root():
    try:
        return jsonify({"message":"hello_world!"})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run()
