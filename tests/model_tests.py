import os
import tensorflow as tf
import app.training.ImageLoader as ld
import app.training.Model as m
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_ds = "~/.certifyai/Columbia_ds"

def load_dataset():
    image_ds = "~/.certifyai/Columbia_ds"
    images = ld.ImageLoader(os.path.expanduser(image_ds))
    images.load()
    return images

def init_model():
    model = m.Model()
    return model
    
def test_image_training_and_validation_datasets_are_loaded():
    images = ld.ImageLoader(os.path.expanduser(image_ds))
    images.load()
    assert images.get_training_dataset() is not None
    assert images.get_validation_dataset() is not None

def test_model_can_be_created():
    print("model can be created")
    model = m.Model()
    assert model.purpose() is not None
    
def test_can_train_model():
    print("test that the model trains")
    tuned_model = init_model().train()
    print(tuned_model.summary())
    # tf.keras.utils.plot_model(tuned_model, show_shapes=True)


def test_can_pass_transform_learning_model():
    input_t = tf.keras.Input(64, 64, 3)
    base_model = tf.python.keras.applications.ResNet50(include_top = False, weights = 'imagenet', input_tensor=input_t)
    
    
    
