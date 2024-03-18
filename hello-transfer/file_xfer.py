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
cg_tensor = tf.convert_to_tensor(cg)

##################################### loads mnist.npz
#%%
mnist = tf.keras.datasets.mnist
# loads the data into a training and test set of features and labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()

##################################### re-save mnist as HW

##################################### merge both npz files.
# %%
