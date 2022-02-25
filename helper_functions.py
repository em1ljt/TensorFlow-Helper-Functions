import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import datetime
import zipfile
import os


def create_model(model_url, num_classes=10, image_shape=(224,224)):
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
    Two figures plotting the loss and accuracy metrics for training and validation 
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
 

def load_and_prep_image(filename, img_shape=224, scale=True):
    '''
    Reads in an image from filename, turns it into a tensor and reshapes
    it into (224, 224, 3)

    Args:
        filename (str): string filename of target image
        img_shape (int): size to reshape the target image to, default 224
        scale (bool): whether to scale the pixel values to range [0, 1], default True
    
    Returns:
        An image as a tensor
    '''
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode the image into a tensor
    img = tf.image.decode_image(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        return img/255
    return img


def pred_and_plot(model, filename, class_names):
    '''
    Imports an image located at filename, makes a prediction on it with a trained model and plots the image with the predicted class as the title.

    Args:
        model: the model to be used for prediction
        filename (str): the name of the image
        class_names (str): the list of classes

    Returms:
        An image with its predicted class as its title
    '''
    # Import the target image and preprocess it
    img = load_and_prep_image(filename)
    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]
    
    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f'Prediction: {pred_class}')
    plt.axis('off')


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10,10), text_size=15, norm=False, savefig=False):
    '''
    Makes a labelled confusion matrix comparing predictions and ground truth labels.
    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
        y_true: Array of truth labels (must be same shape as y_pred).
        y_pred: Array of predicted labels (must be same shape as y_true).
        classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
        figsize: Size of output figure (default=(10, 10)).
        text_size: Size of output figure text (default=15).
        norm: normalize values or not (default=False).
        savefig: save confusion matrix to file (default=False).
    
    Returns:
        A labelled confusion matrix plot comparing y_true and y_pred.
    '''
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalize it
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Find the number of classes
    n_classes = cm.shape[0]

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(
        title='Confusion Matrix',
        xlabel='Predicted Label',
        ylabel='True Label',
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels
    )

    # Make x-axis labels appear on the bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(
                j,
                i,
                f'{cm[i, j]} ({cm_norm[i,j]*100:.1f}%)',
                horizontalalignment='center',
                color='white' if cm[i,j] > threshold else 'black',
                size=text_size
            )
        else:
            plt.text(
                j,
                i,
                f'{cm[i,j]}',
                color='white' if cm[i,j] > threshold else 'black',
                size=text_size
            )
    
    # Save the figure to the current working directory
    if savefig:
        fig.savefig('confusion_matrix.png')


def create_tensorboard_callback(dir_name, experiment_name):
    '''
    Creates a TensorBoard callback instance to store log files.

    Args:
        dir_name (str): target directory to store TensorBoard log files
        experiment_name (str): name of the experiment directory
    
    Returns:
        the TensorBoard callback object
    '''
    log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f'Saving TensorBoard log files to: {log_dir}')
    return tensorboard_callback


def unzip_data(filename):
    '''
    Unzips a file into the current working directory

    Args:
        filename (str): The filepath to the file to be unzipped
    '''
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall()
    zip_ref.close()


def walk_through_dir(dir_path):
    '''
    Walks through dir_path printing out its contents.

    Args:
        dir_path (str): target directory
    '''
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def compare_historys(original_history, new_history, initial_epochs=5):
    '''
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    '''
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def calculate_results(y_true, y_pred):
    '''
    Calculates model accuracy, precision, recall and f1-score of a binary classification model.

    Args:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of a 1D array
    
    Returns:
        A dictionary of accuracy, precision, recall and f1-score
    '''
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1-score using 'weighted average'
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    model_results = {
        'accuracy': model_accuracy,
        'precision': model_precision,
        'recall': model_recall,
        'f1': model_f1
    }
    return model_results