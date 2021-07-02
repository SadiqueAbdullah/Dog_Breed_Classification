#Loading the required libraries
import sys

import cv2
from sklearn.datasets import load_files
from keras.utils import np_utils
import re
import os
import pickle
from glob import glob


from tensorflow import keras
import numpy as np
from tensorflow.python.keras.applications.resnet import ResNet50

from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import io

from keras.applications.resnet import preprocess_input, decode_predictions

def load_dataset(path):
    """
       This function is to load the data from a given location(file-path) and
       data: The data frame containing the images
       dof_files: It contains a dog images from various breed. It is a predictor variable
       dog_targets: Target Variable. Here it's Breed of dog to be classified out of 133 categories.
       """
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets



from pathlib import Path
dataset = Path(os.getcwd())
image_data = Path(os.path.join(dataset,'data/dog_images'))
train_data = Path(os.path.join(dataset,'data/dog_images/train'))
valid_data = Path(os.path.join(dataset,'data/dog_images/valid'))
test_data = Path(os.path.join(dataset,'data/dog_images/test'))
model_test_data = Path(os.path.join(dataset,'test_images'))

# load train, test, and validation datasets
train_files, train_targets = load_dataset(train_data)
valid_files, valid_targets = load_dataset(valid_data)
test_files, test_targets = load_dataset(test_data)

# load list of dog breed names
dog_names = [item[20:-1] for item in sorted(glob("../data/dog_images/train/*/"))]
#load human dataset
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("../../../data/lfw/*/*"))
random.shuffle(human_files)


# Pre-Process the data

from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255




# Obtain Bottle neck features

bottleneck_features = np.load('./data/bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']

#TODO: Define your architecture with Resnet50 network.
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dropout(0.4))
Resnet50_model.add(Dense(133, activation='softmax'))

Resnet50_model.summary()


#Compile the Model with Resnet50 network

Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# Train the Resnet50 Model

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5',
                               verbose=1, save_best_only=True)

Resnet50_model.fit(train_Resnet50, train_targets,
          validation_data=(valid_Resnet50, valid_targets),
          epochs=25, batch_size=30, callbacks=[checkpointer], verbose=1)


# Human detector returns "True" if face is detected in image stored at img_path
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# Dog detector using pretrained Resnet50
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))



#(IMPLEMENTATION) Predict Dog Breed with the Model

### Write a function that takes a path to an image as inputand returns the dog breed that is predicted by the model.

def Resnet50_predict_breed(img_path):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))     # extract bottleneck featurescorresponding to the chosen CNN model.
    predicted_vector = Resnet50_model.predict(bottleneck_feature)       # gets the prediction vector which gives the index of the predicted dog breed.
    return dog_names[np.argmax(predicted_vector)]                       # the o/p is, dog breed that is predicted by the model

#(IMPLEMENTATION) Write your Algorithm

def display_image(img_path):
    image = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imageplot = plt.imshow(cv_rgb)
    return imageplot



def predict_breed(img_path):
    display_image(img_path)
    if dog_detector(img_path):
        print("It's a dog!")
        return print("and it could be from {} Breed of the dog ".format(Resnet50_predict_breed(img_path)))

    if face_detector(img_path):
        print("It's a human!")
        return print("On a funny note ! It resembles to {} Breed of Dog !!".format(Resnet50_predict_breed(img_path)))

    else:
        return print("This seems to be neither dog nor human..It must be something else .")