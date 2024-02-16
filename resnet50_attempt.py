#%%
import matplotlib.pyplot as plotter_lib
import numpy as np
import PIL as image_lib
import tensorflow as tflow
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.initializers import LecunNormal
# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import app.training.ImageLoader as ld
image_ds = "~/.certifyai/Columbia_ds"
img_height = 300
img_width = 300
#%%
images = ld.ImageLoader(os.path.expanduser(image_ds), img_height, img_width)
images.load()
images.get_image_count()

#%%
##### Train model using transfer learning
import keras.models as Model
from keras.layers import Reshape, UpSampling2D, GlobalAveragePooling2D

demo_resnet_model = Sequential()

pretrained_model_for_demo= tflow.keras.applications.ResNet50(include_top=False,
                   input_shape=(img_height, img_width,3),
                   pooling='avg',classes=5,
                   weights='imagenet')

# print(pretrained_model_for_demo.summary())
for each_layer in pretrained_model_for_demo.layers:
        each_layer.trainable=False

### the model
demo_resnet_model.add(pretrained_model_for_demo)
demo_resnet_model.add(Reshape((4, 4, 128)))  # Reshape the output to a 4D tensor
demo_resnet_model.add(UpSampling2D(size=(2, 2)))  # Upsample the output to increase its spatial dimensions
demo_resnet_model.add(Conv2D(filters=2048, 
                             kernel_size=(7, 7), 
                             activation='selu', 
                             kernel_initializer=LecunNormal(),
                             )
                      )
demo_resnet_model.add(GlobalAveragePooling2D())
# demo_resnet_model.add(Conv2D(1024, 
#                              (1, 1), 
#                              activation='selu', 
#                              kernel_initializer=LecunNormal())
#                       )
# demo_resnet_model.add(Conv2D(512, 
#                              (1, 1), 
#                              activation='selu', 
#                              kernel_initializer=LecunNormal())
#                       )
# demo_resnet_model.add(Conv2D(256, 
#                              (1, 1), 
#                              activation='selu', 
#                              kernel_initializer=LecunNormal())
#                       )
# demo_resnet_model.add(Dense(units=128,
#                             activation='selu',
#                             kernel_initializer=LecunNormal())
#                       )
demo_resnet_model.add(Flatten())
demo_resnet_model.add(Dense(2, activation='softmax'))

learning_rate = 1e-5
demo_resnet_model.compile(optimizer=Adam(learning_rate=learning_rate),
                        #   loss='binary_crossentropy',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

epochs=10
history = demo_resnet_model.fit(images.get_training_dataset(), 
                                validation_data=images.get_validation_dataset(), 
                                epochs=epochs
                                )

#%% 
#### Plot the learning model
print(history.history)
plotter_lib.figure(figsize=(8, 8))
epochs_range= range(epochs)
plotter_lib.plot( epochs_range, history.history['accuracy'], label="Training Accuracy")
plotter_lib.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
plotter_lib.axis(ymin=0.4,ymax=1)
plotter_lib.grid()
plotter_lib.title('Model Accuracy')
plotter_lib.ylabel('Accuracy')
plotter_lib.xlabel('Epochs')
plotter_lib.legend(['train', 'validation'])
plotter_lib.show()


# %%        
### Does not work
for image, label in images.get_training_dataset():
            training_labels = label

batch_size = 10
demo_resnet_model.evaluate(images.get_validation_dataset(), training_labels, batch_size=batch_size, verbose=2)


# %%
import app.classifier.ImageClassifier as ic

# create the classifier
classes = ['4cam_auth', '4cam_splc']#images.get_classes() ##["modified", "original"] # should dynamically get this from the model.
classifier = ic.ImageClassifier(demo_resnet_model, classes, img_height, img_width)
#%%
def predict(filename):
    test_image_path = os.path.join("test_images", filename)
    fullpath = os.path.abspath(test_image_path)
    # get a prediction
    score = classifier.predict(filename, fullpath)
    predictionResult = (fullpath, score)

#%%
# give it an image
predict("sitting-monkey-celebs.JPG")
predict("deepfake.jpg")
predict("shawn.jpg")
predict("sitting-monkey.JPG")

# %%
