import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

# The image classifier is where the model is used for the classification of new images.
class ImageClassifier:
    __purpose = ""
    def __init__(self, model, class_names):
        self.__purpose = "decide whether an image is modified or not."
        self.__model = model
        self.__class_names = class_names

    def purpose(self):
        print(self.__purpose)
        
    def classify(self, image):
        # method 2: model + softmax
        probability_model = tf.keras.models.Sequential([
            self.__model,
            tf.keras.layers.Softmax()
        ])
        predictions = probability_model(image)
        pred0 = predictions[0]
        self.print_to_console(pred0)

        self.display_image(pred0, image)
        
    def print_to_console(self, tensor_to_print):
        print("######")
        print(self.__class_names)
        print(tensor_to_print.numpy())

        label0 = np.argmax(tensor_to_print)
        print(self.__class_names[label0])
        print("#####")
        
    # display with jupyter notebook
    def display_image(self, tensor_to_print, image):
        tensor = np.array(image, dtype=np.uint8)
        plt.imshow(tensor[0]) 