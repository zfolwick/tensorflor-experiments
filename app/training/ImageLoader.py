# step 1: load an image dataset and then display it. Be able to pass it around

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

class ImageLoader:
    __data_dir = ""
    __purpose = ""
    def __init__(self, archive):
        self.__purpose = "to pass the sugar"
        self.__data_dir = pathlib.Path(archive).with_suffix('')
        image_count = len(list(self.__data_dir.glob('*/*.jpg'))) # this might be the only thing tying us to images.
        print("loaded " + str(image_count) + " images")
    
    def purpose(self):
        print(self.__purpose)
        
    def displayFirstImage(self, image_directory, i):
        #displays the first in a series of images
        images_in_directory = list(self.__data_dir.glob(image_directory))
        print("found " + str(len(images_in_directory)) + " images")
        img = PIL.Image.open(str(images_in_directory[i]))
        img.show()

    # This loads the image into tensorflow. audio loading also exists.
    def load(self):
        batch_size = 32
        img_height = 180
        img_width = 180

        self.__training_dataset = tf.keras.utils.image_dataset_from_directory(
            self.__data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        self.__class_names = self.__training_dataset.class_names

        self.__validation_dataset = tf.keras.utils.image_dataset_from_directory(
            self.__data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        
        print("finished loading validation and training dataset with classes: " + str(self.__class_names))
    
    def visualize(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.__training_dataset.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                showImage = plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.__class_names[labels[i]])
                plt.axis("off")
                
    def get_training_dataset(self):
        return self.__training_dataset
    
    def get_validation_dataset(self):
        return self.__validation_dataset
    
    def get_classes(self):
        return self.__class_names
    
