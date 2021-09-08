## Dowload packages
!pip install user keras-vggface
!pip install user opencv-python
!pip install user keras-vggface
!pip install keras-vggface
!pip install git+https://github.com/rcmalli/keras-vggface.git
!pip install keras_applications
!pip install kaggle
!pip install keras_vggface
!pip install keras_applications keras_preprocessing

import cv2
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import Model, load_model
from keras.applications.vgg16 import preprocess_input

def build_model(pretrained):
    model = Sequential([
        pretrained,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )
    
    return model

densenet = DenseNet121(
    weights=None,
    include_top=False,
    input_shape=(224,224,3)
)
model = build_model(densenet)
model.summary()

base_path = '/content/drive/MyDrive/Colab Notebooks/'
image_gen = ImageDataGenerator(rescale=1./255.,
                               rotation_range=20,
                               #shear_range=0.2,
                               #zoom_range=0.2,
                               horizontal_flip=True)

train_flow = image_gen.flow_from_directory(
    base_path + 'train/',
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary'
)

image_gen1 = ImageDataGenerator(rescale=1./255.)

valid_flow = image_gen1.flow_from_directory(
    base_path + 'valid/',
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary'
)

train_steps = 8444//64
#valid_steps = 1935//64

history = model.fit_generator(
    train_flow,
    epochs = 5,
    steps_per_epoch= train_steps,
    #validation_data= valid_flow,
    #validation_steps= valid_steps
)

model.save('completed_morphing_v4_augmented_trained_model.h5')

"""
Plot the training and validation loss
epochs - list of epoch numbers
loss - training loss for each epoch
val_loss - validation loss for each epoch
"""
def plot_loss(epochs, loss):
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    #plt.plot(epochs, val_loss, 'orange', label = 'Validation Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
"""
Plot the training and validation accuracy
epochs - list of epoch numbers
acc - training accuracy for each epoch
val_acc - validation accuracy for each epoch
"""
def plot_accuracy(epochs, acc):
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    #plt.plot(epochs, val_acc, 'orange', label = 'Validation accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.show()
    
acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
loss = history.history['loss']
#val_loss = history.history['val_loss']

plot_loss(range(1, len(loss) + 1), loss)
plot_accuracy(range(1, len(loss) + 1), acc)

test_flow = image_gen1.flow_from_directory(
    base_path + 'valid/',
    target_size=(224, 224),
    batch_size=1,
    shuffle = False,
    class_mode='binary'
)
y_pred=model.predict(test_flow)
y_test = test_flow.classes

print("ROC AUC Score:", metrics.roc_auc_score(y_test, y_pred))
print("AP Score:", metrics.average_precision_score(y_test, y_pred))
print()
print(metrics.classification_report(y_test, y_pred > 0.5))