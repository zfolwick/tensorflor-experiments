#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load the training dataset
import app.training.ImageLoader as ld
image_ds = "~/.certifyai/Columbia_ds"
images = ld.ImageLoader(os.path.expanduser(image_ds))
images.load()

####
# QA step for loading images:
#%%
images.visualize(images.get_training_dataset())



#%%
# train the model
import app.training.Model as m
from keras.applications import ResNet50
model = m.Model()
training_set = images.get_training_dataset()
validation_set = images.get_validation_dataset()
model.set_dataset(training_set, validation_set)
resNet_model = ResNet50(weights=None, include_top=False, input_shape=(1200, 1200, 3))
model.set_base(base_model=resNet_model)
batch_size = 16
epochs = 8
tuned_model = model.train(batch_size, epochs)

#%%
#####################
### Plot model
#######################
import tensorflow as tf
print(tuned_model.summary())
tf.keras.utils.plot_model(tuned_model, show_shapes=True)
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


#%%
############################
#### the actual product ####
############################
import app.classifier.ImageClassifier as ic

# create the classifier
classes = ["modified", "original"] # should dynamically get this from the model.
classifier = ic.ImageClassifier(tuned_model, classes)
#%%
def predict(filename):
    test_image_path = os.path.join("test_images", filename)
    fullpath = os.path.abspath(test_image_path)
    # get a prediction
    score = classifier.predict(filename, fullpath)
    predictionResult = (fullpath, score)
#%%
# give it an image
sitting_monkey_celeb = "sitting-monkey-celebs.JPG"
# predict(sitting_monkey_celeb)
shawn = "shawn.jpg"
predict(shawn)
deepfake = "deepfake.jpg"
# predict(deepfake)
# predict("sitting-monkey.JPG")


# %%
#########################################################################################
#### Now we have a probability that something is or is not modified.                 ####
#### This goes into a queue for further humana (and eventually automated) processing.####
#########################################################################################
import app.classifier.PredictionResultHandler as prh
predictionResultHandler = prh.PredictionResultHandler()

predictionResultHandler.handle(predictionResult)
# %%
