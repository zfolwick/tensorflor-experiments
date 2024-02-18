# This takes the mnist dataset and performs an image classification.  This is the "Hello World!" of the machine learning world.  The below script represents the core of the e2e workflow.

# This can be run either as a jupyter notebook or in terminal via python ./mnist_attempt_1.py
# to adjust, scroll down to the banner below:

#  #################################
#  #### Testing
#  #################################
#
# and adjust the index of the x_test array.
#
#%%
import tensorflow as tf
#%%
# data selection
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_test[0])
#normalize
x_train, x_test = x_train / 255.0, x_test / 255.0
#%%
# create architecture
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Softmax()
])

# create model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15)
model.evaluate(x_test,  y_test, verbose=2)

#################################
#### Testing
#################################
# %%
## human testing data selection
img = x_test[1]
expected_label = y_test[1]
print(img)
#%%
# custom data
import os
import numpy as np
filename = "CG-1.jpg"
# test_image_path = os.path.join("test_images", filename)
fullpath = os.path.abspath(filename)
path = tf.keras.utils.get_file(
            filename, "file:\\\\" + fullpath
            )
raw_img = tf.keras.utils.load_img(
            path,
            grayscale=True,
            target_size=(28, 28)
        )
img_array = tf.keras.utils.img_to_array(raw_img)
print("##########################")
print(img_array)
print("##########################")
img = tf.expand_dims(img_array, 0)[0] # Create a batch
expected_label = 'CG'
#%%
# human readable
# print(img)
img = tf.reshape(img, (28, 28))
img = tf.cast(img, dtype=tf.float64)
print(img)
print(img.shape)
# %%
# UNCOMMENT IF YOU WANT TO SEE THIS, OTHERWISE IT'LL STOP THE PROGRAM
from matplotlib import pyplot as plt
plt.imshow(img, interpolation='nearest')
plt.show()
# %%
# actually use the model
import numpy as np
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
print(predictions)
score = tf.nn.softmax(predictions[0])

#assert
print('predicted: ' + str(np.argmax(predictions)))
print('expected: ' + str(expected_label))
# %%
