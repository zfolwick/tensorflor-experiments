import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
from PIL import Image

# The image classifier is where the model is used for the classification of new images.
class ImageClassifier:
    __purpose = ""
    def __init__(self, model, class_names):
        self.__purpose = "decide whether an image is modified or not."
        self.__model = model
        self.__class_names = class_names

    def purpose(self):
        print(self.__purpose)
    
    def predict(self, filename, fullpath):
        path = tf.keras.utils.get_file(
            filename, "file:\\\\" + fullpath
            )
        print(path)

        img = tf.keras.utils.load_img(
            path,
            target_size=(180, 180)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        
        predictions = self.__model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_name = self.__class_names[np.argmax(score)]
        self.print_to_console(class_name, score)
        self.display_image(path)

        
    def print_to_console(self, class_name, score):
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_name, 100 * np.max(score))
        )
        
    def display_image(self, filepath):
        img = Image.open(filepath)
        plt.imshow(img)