# 

#%%
import tensorflow as tf
import keras_tuner as kt
import os
import numpy as np
from matplotlib import pyplot as plt

####################
#### load model ####
####################
# %%
if os.path.exists("mnist_model"):
  old_model = tf.keras.models.load_model("mnist_model")

## Display model layers
#%%
def print_model(model):
    model.summary()
    print(len(model.layers))
    tf.keras.utils.plot_model(
        model,
        show_shapes=True,
        show_layer_activations=True,
        expand_nested=True,
        rankdir='TB'
        )

print_model(old_model)
#######################
#### freeze layers ####  
#######################
from keras.models import Model
old_model.trainable = False
#strip flatten and final classification layer
old_model = Model(old_model.input, old_model.layers[-4].output)
base_model = tf.keras.Sequential() # Create a new model from the 2nd layer and all the convolutional blocks
for layer in old_model.layers[1:]:
  base_model.add(layer)