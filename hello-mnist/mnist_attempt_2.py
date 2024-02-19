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
#################################
#### Model creation library
#################################
import tensorflow as tf

def data_selection():
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  #normalize
  x_train, x_test = x_train / 255.0, x_test / 255.0
  return (x_train, y_train), (x_test, y_test) 

def create_model(test, train, labels):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
  ])

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

  model.fit(train, labels, epochs=5)
  model.evaluate(test,  labels, verbose=2)
  
  return model



#%%
#################################
#### Testing functions Library
#################################
import os
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(linewidth=180)
## human testing data selection
def use_data_from_training_set(index):
  img = x_test[index]
  expected_label = y_test[index]
  return img, expected_label


def get_data_from_file(filename, label):
  fullpath = os.path.abspath(filename)
  # path = tf.keras.utils.get_file(
  #           filename, "file://" + fullpath
  #           )
  raw_img = tf.keras.utils.load_img(
              fullpath,
              grayscale=True,
              target_size=(28, 28)
          )
  img_array = tf.keras.utils.img_to_array(raw_img)
  img = tf.expand_dims(img_array, 0)[0] # Create a batch
  return img

# UNCOMMENT IF YOU WANT TO SEE THIS, OTHERWISE IT'LL STOP THE PROGRAM
def show_img(img):
  img = tf.reshape(img, (28, 28))
  img = tf.cast(img, dtype=tf.float64)
  print(img)
  img_array = tf.keras.utils.img_to_array(img)
  img = tf.expand_dims(img_array, 0)[0] # Create a batch
  plt.imshow(img, interpolation='nearest')
  plt.show()

def predict(model, img_array):
  img_array = tf.expand_dims(img_array, 0) # Create a batch
  predictions = model.predict(img_array, verbose=None)
  score = tf.nn.softmax(predictions[0])
  # print(score)
  return predictions

def test_jpg(filename, expected_value):
  img_array = get_data_from_file(filename=filename, label=f'CG-{expected_value}')
  predictions = predict(model=model, img_array=img_array)
  equal = np.argmax(predictions) == expected_value

  print(f'predicted value: {str(np.argmax(predictions))} : expected: {str(expected_value)} : equal: {str(equal)}')
  # show_img(img=img_array) # uncomment this if you run as a jupyter notebook
  return 1 if equal else 0

def test_digits(directory_name, extension):
  print(f'testing {directory_name}')
  current_score = 0
  max_glyphs = 10
  for red_idx in range(max_glyphs):
    expected_value = red_idx
    score = test_jpg(filename=f"{directory_name}/CG-{expected_value}.{extension}", expected_value=expected_value)
    current_score += score
  
  print(f"Test of {directory_name} completed.  Score is: {current_score}/{max_glyphs}")





####################################################################
####################################################################
####################################################################
##############         USAGE
####################################################################
####################################################################
####################################################################

#%%
import sys
# model creation/training
if len(sys.argv) > 1 and sys.argv[1] == "create":
  (x_train, y_train), (x_test, y_test) = data_selection()
  model = create_model(x_test, x_test, y_test)
  model.save("mnist_model")

# %%
if os.path.exists("mnist_model"):
  model = tf.keras.models.load_model("mnist_model")
# actual model testing
test_digits('Red', 'jpg')
test_digits('Blue-on-white', 'jpg')
test_digits('White', 'jpg')
test_digits('PNG/black-on-white', 'png')
test_digits('PNG/white-on-black', 'png')
test_digits('white-background', 'jpg')
