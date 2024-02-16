# Create the model from a dataset.
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.initializers import LecunNormal
import pathlib
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.optimizers import Adam


class Model:

    def __init__(self):
        self.__purpose = None
        self.__training_images = None
        self.__model = None
        self.__training_set = None
        self.__validation_set = None
        self.__history = None
        self.__purpose = "Create the model from a dataset and train it."
        self.purpose()
        

    def purpose(self):
        print(self.__purpose)
    
    def set_base(self, base_model):
        self.__model = base_model
    
    def set_dataset(self, training_set, validation_set):
        self.__training_set = training_set
        self.__validation_set = validation_set
        
    def train(self, batch_size, epochs, learning_rate):
        num_classes = 2
        model = self.create(num_classes=num_classes, base_model=self.__model)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy'])
        
        # batch_size=10 # how many pics at once?
        # epochs=8 
             
        for image, label in self.__training_set:
            self.__training_images = image
            training_labels = label

        
        history = model.fit(self.__training_images, 
                            training_labels, 
                            batch_size=batch_size, 
                            epochs=epochs,
                            verbose=2,
                            validation_split=0.2,
                            validation_data=self.__validation_set, 
                            shuffle=True)
        # # Plot training and validation accuracy - not actually working
        # plt.plot(history.history['accuracy'], label='Training Accuracy')
        # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.title('Training and Validation Accuracy')
        # plt.legend()
        # plt.show()
        self.__history = history     
        model.evaluate(self.__validation_set, training_labels, batch_size=batch_size, verbose=2)
        
        return model

    def create(self, num_classes, base_model):
        
        # # Load the pre-trained ResNet50 model without the top layers

        # # Freeze the layers in the base model (optional)
        if base_model is not None:
            for layer in base_model.layers:
                layer.trainable = False
        
        to_res = (1200, 1200)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
        if base_model is not None:
            model.add(base_model)
        model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), kernel_initializer=LecunNormal(), activation='selu', input_shape=(1200,1200,3)))
        # model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.Dense(256, activation='relu'))
        # model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Dense(128, activation='relu'))
        # model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Dense(64, activation='relu'))
        # model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            
        return model
        
    def get_history(self):
        return self.__history.history
    
    def get_training_images_tensor(self):
        return self.__training_images
        