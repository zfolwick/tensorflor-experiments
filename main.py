#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import app.training.ImageLoader as ld
images = ld.ImageLoader("./image-library")
images.load()

# now we get to image classification/NN stuff.
import app.training.Model as m
model = m.Model(images.get_training_dataset(), images.get_validation_dataset())
tuned_model = model.train()

# persist the model
tuned_model.save("my_first_model")
print("Completed writing model!")
#%%


#%%
import app.classifier.ImageClassifier as ic
import tensorflow as tf
# load model
loaded_model = tf.keras.models.load_model("my_first_model")

# give it a candidate image.
classes = images.get_classes() # should dynamically get this from the model.
classifier = ic.ImageClassifier(tuned_model, classes)
classifier.classify(model.get_training_images_tensor())
# %%