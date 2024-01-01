#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ImageLoader as ld
images = ld.ImageLoader("../image-library")
images.load()

# now we get to image classification/NN stuff.
import Model as m
model = m.Model(images.get_training_dataset(), images.get_validation_dataset())
tuned_model = model.train()

import ImageClassifier as ic
classifier = ic.ImageClassifier(tuned_model, images.get_classes())
classifier.classify(model.get_training_images_tensor())
# %%