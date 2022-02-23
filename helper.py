import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np


def create_model(model_url, num_classes=10, image_shape):
  '''
  Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in the output layer,
      should be equal to number of target classes, default 10.
  
  Returns:
    An uncompiled Keras Sequential model with model_url as feature extractor
    layer and Dense output layer with num_classes output neurons.
  '''
  # Download the pretrained model and save it as a Keras layer
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=False, # freeze the already learnt patterns
                                           name='feature_extraction_layer',
                                           input_shape=image_shape+(3,)) # (224,224,3)
  
  # Create our own model
  model = tf.keras.Sequential([
      feature_extractor_layer,
      tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')
  ])

  return model


def plot_loss_curves(history):
  '''
  Returns separate loss curves for training and validation metrics.
  
  Args:
    history (History object): A TensorFlow History object
    
  Returns:
    Two figures plotting the loss and accuracy metrics for training and
    validation 
  '''
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='Training Loss')
  plt.plot(epochs, val_loss, label='Validation Loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='Training Accuracy')
  plt.plot(epochs, val_accuracy, label='Validation Accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend()
 
