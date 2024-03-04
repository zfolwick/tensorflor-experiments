# load the existing model
# perform standard transfer learning
# use binary crossEntropy: 0 is handwritten, 1 is computer generated
# it makes sense to have the output layer be Dense(1, activation='sigmoid'), and the
#    previous layer correspond to all the different ways computer generated lines can exist
# I expect there to be at least 1 hidden layer, with an unknown number of units.  Use keras
#    tuner to discover the correct number of nodes. 
#%%
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#%%
#####################
#### fetch model ####
#####################
old_model = tf.keras.models.load_model("../hello-mnist/mnist_model")
#%%
old_model.summary()
print(len(old_model.layers))
tf.keras.utils.plot_model(
    old_model,
    show_shapes=True,
    show_layer_activations=True,
    expand_nested=True,
    rankdir='TB'
    )


#%%
# augment the existing data with new  data
from keras.preprocessing.image import ImageDataGenerator
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
# Use the correct dataset
# Red, PNG/red-on-black-sharp-antialiasing, and PNG/red-on-black-crisp-antialiasing completed seem to do very well
import numpy as np
from keras.datasets import mnist
ds_CG = tf.keras.utils.image_dataset_from_directory(
    directory="CG-images",
    labels="inferred",
    color_mode="grayscale",
    image_size=(28,28)
    )

(x_train_mnist, mnist_number_training_labels), (_, _) = mnist.load_data()
#%%
print('########')
print(ds_CG)
print(x_train_mnist.shape)
print('########')
#%%
#### Make the custom dataset the same shape:
#%%
# Convert custom dataset to NumPy arrays

custom_images = []
augmented_ds_CG = []
for images, labels in ds_CG:
    cg_labels = np.array(["CG"] * len(images))
    hw_labels = np.array(["HW"] * len(x_train_mnist))
    custom_images.append(images.numpy())

# Convert custom dataset to NumPy arrays
custom_images = np.array(custom_images)
custom_labels = np.array(cg_labels)

#%%
# Reshape the MNIST data to match the shape of ds_CG
print('##################')
print(x_train_mnist.shape)
print(custom_images.shape)

x_train_mnist = np.expand_dims(x_train_mnist, axis=-1)
# x_train_mnist = np.squeeze(x_train_mnist, axis=(3,4))
custom_images = np.squeeze(custom_images, axis=(0))

print('#####  NEW  #############')
print(x_train_mnist.shape)
print(custom_images.shape)


#%%
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



# Concatenate augmented images into a single array
augmented_images = np.concatenate(augmented_images, axis=0)

augmented_labels = np.full(len(augmented_images),"CG")
# Print the shape of the augmented images array
print("Shape of augmented images:", augmented_images.shape)
print(len(augmented_images))
print(len(augmented_labels))






#%%
# Concatenate the data
# comine the images: cg image, hw images
x_train_combined = np.concatenate((augmented_images, x_train_mnist), axis=0)
y_train_combined = np.concatenate((augmented_labels, hw_labels), axis=0)

# Shuffle the combined dataset
shuffle_index = np.random.permutation(len(x_train_combined))
x_train_combined = x_train_combined[shuffle_index]
y_train_combined = y_train_combined[shuffle_index]


# Print the shape of the combined dataset
print("Combined training dataset shape:", x_train_combined.shape)

#%%
###############################
####### XFER
###############################
from keras.models import Model

old_model.trainable = False
#strip flatten and final classification layer
old_model = Model(old_model.input, old_model.layers[-4].output)

base_model = tf.keras.Sequential() # Create a new model from the 2nd layer and all the convolutional blocks
for layer in old_model.layers[1:]:
  base_model.add(layer)

#%%
model = tf.keras.Sequential()
model.add(base_model)
# try to recognize shapes
model.add(tf.keras.layers.Conv2D(filters=32, input_shape=(28,28,1), kernel_size=5, activation='selu'))
model.add(tf.keras.layers.Conv2D(filters=16, activation='selu', kernel_size=3))

model.add(tf.keras.layers.Flatten())
# get a ton of neurons... we'll whittle this down later.  
model.add(tf.keras.layers.Dense(units=1024, activation='selu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#%%
# loss function is changed because there's only 2 classes
loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(
    optimizer='adam', # probably will change this to some L2 error or something
    loss=loss_function,
    metrics=['accuracy'])

batch_size=32 # how many pics at once?
epochs=15  
# undefined variables alert
#%%
combined_dataset = tf.data.Dataset.from_tensor_slices((x_train_combined, y_train_combined))

# Get the total number of samples
num_samples = len(x_train_combined)

# Calculate the number of samples for training and testing
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size

# Shuffle and split the dataset
train_dataset = combined_dataset.take(train_size)
test_dataset = combined_dataset.skip(train_size)
# Print the number of samples in each split
print("Number of samples in training set:", train_size)
print("Number of samples in testing set:", test_size)
#%%
# Fit the model
model.fit(train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE),
          epochs=epochs,
          validation_data=test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))
    
##########################################
#######  Testable assertions 
##########################################
# 1. the text in PNG/red-on-black-crisp-antialiasing is determined to be non-handwritten
#    a. PNG/red-on-black-crisp-antialiasing/CG-0.png is NOT handwritten
#    b. PNG/red-on-black-crisp-antialiasing/CG-1.png is NOT handwritten
#    c. PNG/red-on-black-crisp-antialiasing/CG-2.png is NOT handwritten
#    d. PNG/red-on-black-crisp-antialiasing/CG-3.png is NOT handwritten
#    e. PNG/red-on-black-crisp-antialiasing/CG-4.png is NOT handwritten
#    f. PNG/red-on-black-crisp-antialiasing/CG-5.png is NOT handwritten
#    g. PNG/red-on-black-crisp-antialiasing/CG-6.png is NOT handwritten
#    h. PNG/red-on-black-crisp-antialiasing/CG-7.png is NOT handwritten
#    i. PNG/red-on-black-crisp-antialiasing/CG-8.png is NOT handwritten
#    j. PNG/red-on-black-crisp-antialiasing/CG-9.png is NOT handwritten
# 2. the text in directories PNG, Red, White, blue-on-white is NOT handwritten
# 3. pics of handwritten digits are determined to be handwritten
# 4. handwritten digits from the dataset are determined to be handwritten
# %%
