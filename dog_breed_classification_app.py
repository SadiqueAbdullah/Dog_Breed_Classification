from _ast import Import
from sklearn.datasets import load_files
from keras.utils import np_utils
# Importing required libraries_Udacity
import cv2
import numpy as np
from glob import glob
from flask import Flask
from tensorflow import keras
import numpy as np
import pandas as pd
# import the necessary packages
from tensorflow.python.keras.applications.resnet import ResNet50
#from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import flask
import io
import os
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None


# defining the Model
def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = ResNet50(weights="imagenet")

# Image preparation and Processing

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

        # resize the input image and preprocess it
        image = image.resize(target)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
#
# def display_image(img_path):
#     image = cv2.imread(img_path)
#     cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     imageplot = plt.imshow(cv_rgb)
#     return imageplot
#
# def predict_breed(img_path):
#     display_image(img_path)
#     if dog_detector(img_path):
#         print("It's a dog!")
#         return print("and it could be from {} Breed of the dog ".format(Resnet50_predict_breed(img_path)))
#
#     if face_detector(img_path):
#         print("It's a human!")
#         return print(
#             "On a funny note ! It resembles to {} Breed of Dog !!".format(Resnet50_predict_breed(img_path)))
#
#     else:
#         return print("This seems to be neither dog nor human..It must be something else .")
# return the processed image
        return image

#Frame work

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()


 # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # load the model
            model = ResNet50(weights="imagenet")

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

#Starting your Keras Rest API

##We can use curl to pass this image to our API and find out what ResNet thinks the image contains:


{
  "predictions": [
    {
      "label": "beagle",
      "probability": 0.9901360869407654
    },
    {
      "label": "Walker_hound",
      "probability": 0.002396771451458335
    },
    {
      "label": "pot",
      "probability": 0.0013951235450804234
    },
    {
      "label": "Brittany_spaniel",
      "probability": 0.001283277408219874
    },
    {
      "label": "bluetick",
      "probability": 0.0010894243605434895
    }
  ],
  "success": true
}

#Consuming the Keras REST API programmatically

# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "dog.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"],
            result["probability"]))

# otherwise, the request failed
else:
    print("Request failed")