# Create the model from a dataset.
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

class Model:
    __purpose = ""
    def __init__(self, training_set, validation_set):
        self.__purpose = "Create the model from a dataset and train it."
        self.purpose()
        self.__training_set = training_set
        self.__validation_set = validation_set
        
    def purpose(self):
        print(self.__purpose)
        
    def train(self):
        num_classes = 2
    
        model = self.create(num_classes)
        
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        
        batch_size=64 # how many pics at once?
        epochs=5   
             
        for image, label in self.__training_set:
            self.__training_images = image
            training_labels = label

        
        model.fit(self.__training_images, training_labels, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)
        model.evaluate(self.__training_images, training_labels, batch_size=batch_size, verbose=2)
        
        return model

    def create(self, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes),
            # for the imageclassifier to be better able to handle it:
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(128, activation='relu'),
            # 2 different classes
            tf.keras.layers.Dense(2),
        ])
        
        return model
        
    def get_training_images_tensor(self):
        return self.__training_images
        