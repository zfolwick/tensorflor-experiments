#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load the training dataset
import app.training.ImageLoader as ld
images = ld.ImageLoader("./image-library")
images.load()

#%%
# train the model
import app.training.Model as m
model = m.Model(images.get_training_dataset(), images.get_validation_dataset())
tuned_model = model.train()

# persist the model
tuned_model.save("my_first_model")
print("Completed writing model to disk!")

#%%
#####################
#### fetch model ####
#####################
import tensorflow as tf
# load model
loaded_model = tf.keras.models.load_model("my_first_model")
# loaded_model.summary()
##%%
############################
#### the actual product ####
############################
import app.classifier.ImageClassifier as ic

# create the classifier
classes = ["modified", "original"] # should dynamically get this from the model.
classifier = ic.ImageClassifier(loaded_model, classes)
##%%
# give it an image
filename = "sitting-monkey.JPG"
test_image_path = os.path.join("test_images", filename)
fullpath = os.path.abspath(test_image_path)
# get a prediction
score = classifier.predict(filename, fullpath)

predictionResult = (fullpath, score)
## %%
#########################################################################################
#### Now we have a probability that something is or is not modified.                 ####
#### This goes into a queue for further humana (and eventually automated) processing.####
#########################################################################################
import app.classifier.PredictionResultHandler as prh
predictionResultHandler = prh.PredictionResultHandler()

predictionResultHandler.handle(predictionResult)
# %%
