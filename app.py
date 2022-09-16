# Importing required packages
import os
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, url_for, render_template

# Load model
model_path = 'model/image_classifier_model.h5'
pretrained_model = tf.keras.models.load_model(model_path)

# Create flask application
app = Flask(__name__)


def make_predictions(image_name, image_size=(180, 180)):
    # check if upload directory exits
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # load & process image
    img = keras.preprocessing.image.load_img(
        os.path.join('uploads', image_name),
        target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    # make predictions
    predictions = pretrained_model.predict(img_array)
    score = predictions[0]

    return (
        "This image is %.2f percent cat and %.2f percent dog."
        % (100 * (1 - score), 100 * score)
    )


@app.route('/')
def index():
    return render_template('image.html')


@app.route('/api/image', methods=['POST'])
def upload_image():
    # check for image
    if 'image' not in request.files:
        return render_template('image.html', prediction='No image uploaded!!!')
    file = request.files['image']
    if file.filename == '':
        return render_template('image.html', prediction='No image selected!!!')
    file_name = secure_filename(file.filename)
    file.save(os.path.join('uploads', file_name))

    # make predictions
    predictions = make_predictions(file_name)

    return render_template('image.html', prediction=predictions)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
