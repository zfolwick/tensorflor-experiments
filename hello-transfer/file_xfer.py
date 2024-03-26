####################### creates an npz file from CG data
#%%
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

#%%
# augment the existing data with new  data
datagen = ImageDataGenerator(
    rotation_range=20,      # Random rotation in the range [-20, 20] degrees
    width_shift_range=0.1,  # Random horizontal shift by 10% of the width
    height_shift_range=0.1, # Random vertical shift by 10% of the height
    shear_range=0.2,        # Random shear transformation in the range [-0.2, 0.2]
    zoom_range=0.2,         # Random zoom by scaling up to 20%
    horizontal_flip=True,   # Random horizontal flip
    fill_mode='nearest'     # Fill mode for handling newly created pixels
)
#%%
# Convert custom dataset to NumPy arrays
ds_CG = tf.keras.utils.image_dataset_from_directory(
    directory="CG-images",
    labels="inferred",
    color_mode="grayscale",
    image_size=(28,28)
    )

custom_images = []
augmented_ds_CG = []
for images, labels in ds_CG:
    cg_labels = np.array(["CG"] * len(images))
    custom_images.append(images.numpy())

# Convert custom dataset to NumPy arrays
custom_images = np.array(custom_images)
custom_labels = np.array(cg_labels)

#%%

custom_images = np.squeeze(custom_images, axis=0)

print('#####  NEW  #############')
print(custom_images.shape)

#### Augment the custom data
# Augment the images
desired_augmented_samples = 60000
augmented_images = []
augmented_labels = []
print(len(custom_images))
for batch in datagen.flow(custom_images, batch_size=len(custom_images), shuffle=False):
    augmented_images.append(batch)
    if len(augmented_images) * len(batch) >= desired_augmented_samples:
        break

print(f'augmented images: {type(augmented_images)}')

# Concatenate augmented images into a single array
augmented_images = np.concatenate(augmented_images, axis=0)

augmented_labels = np.full(len(augmented_images),"CG")
#%%
# Print the shape of the augmented images array
print("Shape of augmented images:", augmented_images.shape)
print(len(augmented_images))
print(type(augmented_images))
print(len(augmented_labels))

#%%
# save as an npz file
np.savez_compressed('CG-images/dataset/cg/computer_generated.npz',features=augmented_images, label="CG" )
##################################### loads cg.npz
#%%
cg_npz_file = 'CG-images/dataset/cg/computer_generated.npz'
cg = np.load(cg_npz_file)
##################################### re-save mnist as HW

#%%
mnist = tf.keras.datasets.mnist
# loads the data into a training and test set of features and labels
(hw_images, _), (_, _) = mnist.load_data()
hw_npz_file = 'CG-images/dataset/hw/handwritten.npz'
np.savez_compressed(hw_npz_file, features=hw_images, label='HW')

##################################### loads handwritten data
hw = np.load(hw_npz_file)

##################################### merge both npz files.
# %%
# merge them both
# Concatenate features and labels
features = np.concatenate((tf.squeeze(cg["features"].astype(np.uint8))
, hw["features"]), axis=0)
labels = np.concatenate((np.zeros(len(cg["features"])), np.ones(len(hw["features"]))), axis=0)

#%%
from sklearn.model_selection import train_test_split
# Split the data into training, validation, and test sets
images_train, images_test, images_labels_train, images_labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# %%
# prove to yourself
#  that it's really correct
#  by displaying the image
from matplotlib import pyplot as plt
# UNCOMMENT IF YOU WANT TO SEE THIS, OTHERWISE IT'LL STOP THE PROGRAM
def show_img(img):
  img = tf.reshape(img, (28, 28))
  img = tf.cast(img, dtype=tf.float64)
  # print(img)
  img_array = tf.keras.utils.img_to_array(img)
  img = tf.expand_dims(img_array, 0)[0] # Create a batch
  plt.imshow(img, interpolation='nearest')
  plt.show()
  
first_image = images_train[0]
show_img(first_image)








# %%
import sys
import keras_tuner as kt
##########################################################
##########################################################
# You've successfully merged both datasets.            ###
#  Now you need to perform the image classification    ###
##########################################################
##########################################################

def model_builder(hp):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=10)
  for i in range(hp.Int('n_layers', 1, 10)):
    model.add(tf.keras.layers.Dense(units=hp_units))
    # model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Activation('gelu'))
  
  model.add(tf.keras.layers.Dense(units=100))

  print(model.summary())

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

# if len(sys.argv) > 1 and sys.argv[1] == "create":

tuner = kt.Hyperband(model_builder,
                    objective='val_accuracy',
                    max_epochs=10,
                    factor=2,
                    directory='my_new_dir',
                    project_name='intro_to_cg')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(images_train, images_labels_train, epochs=5, validation_split=0.2, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=3)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
model = tuner.hypermodel.build(best_hps)
history = model.fit(images_train, images_labels_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(images_train, images_labels_train, epochs=best_epoch, validation_split=0.2)
eval_result = hypermodel.evaluate(images_train, images_labels_train)
print("[test loss, test accuracy]:", eval_result)

model.save("fake_vs_mnist_model")



#%%
##############################################################
####  TESTING ################################################
##############################################################
# We provide several images of CG images and several of HW images.
#  This should work not only on numbers but letters as well, without extra work.
#  Need a set of simple handwritten letters and numbers, and electronically generated letters and numbers.
#  should this work on standard mnist arabic letters or numbers (see: https://numpy-datasets.readthedocs.io/en/latest/modules/images.html), although much pre-processing will be required to create a 28 by 28 images.
#
##### TEST PLAN
## 1. Test that typed digits NOT USED FOR DATA GENERATION are classified as computer generated images
## 2. Test that typed letters are classified as computer generated images
## 3. Test that fashion mnist pictures are classified as computer generated images
## 4. Test that handwritten letters are classified as handwritted letters
## 5. Test that hand drawn sketches of 28x28 are classified as handwritten.
## 6. Test that typewriter created characters that are scanned are classified as handwritten (they are not fake, technically) - this should not be blocking, but it could describe a whole class of behavior.

#%%
################################################################################
######  TESTING ASSETS #########################################################
################################################################################
import os
def get_data_from_file(filename):
  fullpath = os.path.abspath(filename)

  raw_img = tf.keras.utils.load_img(
              fullpath,
              color_mode="grayscale",
              target_size=(28, 28)
          )
  img_array = tf.keras.utils.img_to_array(raw_img)
  img = tf.expand_dims(img_array, 0)[0] # Create a batch
  return img


def predict(model, img_array):
  img_array = tf.expand_dims(img_array, 0) # Create a batch
  predictions = model.predict(img_array, verbose=None)
  score = tf.nn.softmax(predictions[0])
  # print(score)
  # print(predictions)
  return predictions

def test_image(filename, expected_value):
  img_array = get_data_from_file(filename=filename)
  show_img(img_array)
  predictions = predict(model=model, img_array=img_array)
  predicted_value = np.argmax(predictions)
  print(f'predicted: {Classification(predicted_value)}')
  print(f'expected: {expected_value}')
  equal = predicted_value == expected_value.value
  return equal

def test_digits(directory_name, partial_file_name, extension, expected_value):
  print(f'testing {directory_name}')
  current_score = 0
  max_glyphs = 10
  for red_idx in range(max_glyphs):
    score = test_image(filename=f"{directory_name}/{partial_file_name}{red_idx}.{extension}", expected_value=expected_value)
    current_score += score
  
  print(f"Test of {directory_name} completed.  Score is: {current_score}/{max_glyphs}")

from enum import Enum
class Classification(Enum):
  CG = 0
  HW = 1
  
#%%
cg_test_file_base = 'test-images/computer-generated'
file_under_test = f'{cg_test_file_base}/PNG/red-on-black-crisp-antialiasing/CG-0.png'
test_image(filename=file_under_test, expected_value=Classification.CG) == True
file_under_test = f'{cg_test_file_base}/PNG/red-on-black-crisp-antialiasing/CG-1.png'
test_image(filename=file_under_test, expected_value=Classification.CG) == True
file_under_test = f'{cg_test_file_base}/PNG/red-on-black-crisp-antialiasing/CG-2.png'
test_image(filename=file_under_test, expected_value=Classification.CG) == True
file_under_test = f'{cg_test_file_base}/PNG/red-on-black-crisp-antialiasing/CG-3.png'
test_image(filename=file_under_test, expected_value=Classification.CG) == True
file_under_test = f'{cg_test_file_base}/PNG/red-on-black-crisp-antialiasing/CG-4.png'
test_image(filename=file_under_test, expected_value=Classification.CG) == True
#%%
hw_test_file_base = 'test-images/handwritten-numbers'
file_type = 'Ballpoint'
file_under_test = f'{hw_test_file_base}/{file_type}/0.jpg'
test_image(filename=file_under_test, expected_value=Classification.HW) == True
file_under_test = f'{hw_test_file_base}/{file_type}/1.jpg'
test_image(filename=file_under_test, expected_value=Classification.HW) == True
file_under_test = f'{hw_test_file_base}/{file_type}/2.jpg'
test_image(filename=file_under_test, expected_value=Classification.HW) == True
file_under_test = f'{hw_test_file_base}/{file_type}/3.jpg'
test_image(filename=file_under_test, expected_value=Classification.HW) == True
file_under_test = f'{hw_test_file_base}/{file_type}/8.jpg'
test_image(filename=file_under_test, expected_value=Classification.HW) == True

#%%
file_type = 'Sharpie'
file_under_test = f'{hw_test_file_base}/{file_type}/0.jpg'
test_image(filename=file_under_test, expected_value=Classification.HW) == True
file_under_test = f'{hw_test_file_base}/{file_type}/1.jpg'
test_image(filename=file_under_test, expected_value=Classification.HW) == True
file_under_test = f'{hw_test_file_base}/{file_type}/2.jpg'
test_image(filename=file_under_test, expected_value=Classification.HW) == True
file_under_test = f'{hw_test_file_base}/{file_type}/3.jpg'
test_image(filename=file_under_test, expected_value=Classification.HW) == True
file_under_test = f'{hw_test_file_base}/{file_type}/8.jpg'
test_image(filename=file_under_test, expected_value=Classification.HW) == True
# %%
test_digits(directory_name=f'{cg_test_file_base}/PNG/red-on-black-crisp-antialiasing', 
            partial_file_name='CG-', 
            extension='png',
            expected_value=Classification.CG)
test_digits(directory_name=f'{cg_test_file_base}/PNG/red-on-black-sharp-antialiasing', 
            partial_file_name='CG-',
            extension='png',
            expected_value=Classification.CG)
test_digits(directory_name=f'{cg_test_file_base}/PNG/red-on-black-smooth-antialiasing', 
            partial_file_name='CG-', 
            extension='png',
            expected_value=Classification.CG)
test_digits(directory_name=f'{cg_test_file_base}/PNG/red-on-black-strong-antialiasing', 
            partial_file_name='CG-', 
            extension='png',
            expected_value=Classification.CG)
# %%
