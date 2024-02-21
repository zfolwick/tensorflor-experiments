# This takes the mnist dataset and performs an image classification. 
# This is the "Hello World!" of the machine learning world.  
# The below script represents the core of the e2e workflow, encapsulated in methods to
# allow for code reuse.  
# 
# The model is trained using keras Tuner. To install, use: pip install -q -U keras-tuner

#
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
import keras_tuner as kt

def model_builder(hp):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  for i in range(hp.Int('n_layers', 1, 10)):
    model.add(tf.keras.layers.Dense(units=hp_units))
    # model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Activation('gelu'))
  
  model.add(tf.keras.layers.Dense(units=500))

  print(model.summary())

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

def data_selection():
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  #normalize
  x_train, x_test = x_train / 255.0, x_test / 255.0
  return (x_train, y_train), (x_test, y_test) 

def create_model(test, train, labels):
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
  
  #hidden layers
  model.add(tf.keras.layers.Ac)
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.2))
  
  # 
  model.add(tf.keras.layers.Dense(500)) # number of classes.
  model.add(tf.keras.layers.Softmax()) # use softmax when having multiple classes

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizer.Adam()
  model.compile(optimizer=optimizer,
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
  # return 1 if equal else show_img(img=img_array) # uncomment this if you run as a jupyter notebook
  return 1 if equal else 0 # uncomment this if you run as a jupyter notebook


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
  
  tuner = kt.Hyperband(model_builder,
                      objective='val_accuracy',
                      max_epochs=10,
                      factor=3,
                      directory='my_dir',
                      project_name='intro_to_kt')
  stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
  tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
  best_hps=tuner.get_best_hyperparameters(num_trials=3)[0]

  print(f"""
  The hyperparameter search is complete. The optimal number of units in the first densely-connected
  layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
  is {best_hps.get('learning_rate')}.
  """)
  model = tuner.hypermodel.build(best_hps)
  history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

  val_acc_per_epoch = history.history['val_accuracy']
  best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
  print('Best epoch: %d' % (best_epoch,))
  
  hypermodel = tuner.hypermodel.build(best_hps)
  hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)
  eval_result = hypermodel.evaluate(x_test, y_test)
  print("[test loss, test accuracy]:", eval_result)

  # model = create_model(x_test, x_test, y_test)
  model.save("mnist_model")

# %%
if os.path.exists("mnist_model"):
  model = tf.keras.models.load_model("mnist_model")
# actual model testing
test_digits('Red', 'jpg')
test_digits('White', 'jpg')
test_digits('PNG/red-on-black-crisp-antialiasing', 'png')
test_digits('PNG/red-on-black-sharp-antialiasing', 'png')
test_digits('PNG/red-on-black-smooth-antialiasing', 'png')

# test_digits('white-background', 'jpg')
# test_digits('PNG/White-on-black', 'png')
# test_digits('Blue-on-white', 'jpg')
# test_digits('PNG/red-on-black-strong-antialiasing', 'png')
# %%
