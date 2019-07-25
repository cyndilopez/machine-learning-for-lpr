from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.applications.nasnet import NASNetLarge, preprocess_input
# from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight

# keras.applications.mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import random

pathname = 'expanded_transflearn_database'
image_width = 224
image_height = 224
train_dir = 'expanded_transflearn_database'
batch_size = 16

resnet_weights_path = 'imagenet'

clf = Sequential()

image_files = []
TRACES = os.listdir(pathname)
for jj, folder in enumerate(TRACES):
        if not folder.startswith('.'):
                for ii, trace in enumerate(os.listdir(pathname+'/'+folder)):
                        if not trace.startswith('.'):
                                image_files.append(pathname+'/'+folder+'/' + trace)

rand_sample = random.sample(range(len(image_files)),5)

sample_image_files = []
for i in rand_sample:
        sample_image_files.append(image_files[i])

X = np.zeros((len(sample_image_files), image_height, image_width, 3))
for index, image_path in enumerate(sample_image_files):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(image_height, image_width))
    img = img.reshape((image_height, image_width, 3))
    X[index] = img




# 1st layer as the lumpsum weights from 'imagenet'
# this layer is set up as not trainable, use it as is
clf.add(ResNet50(include_top=False,
                weights=resnet_weights_path,
                input_shape=(image_height,image_width,3),
                pooling='avg'))



#2nd layer as Dense for classifcation

# train_datagen = ImageDataGenerator(
#         preprocessing_function=preprocess_input,
#         zca_whitening=True,
#         validation_split=0.2
# )

# train_datagen.fit(X)

train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        zca_whitening=True,
        validation_split=0.2
)


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(image_height, image_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training')
                        

print(train_generator.class_indices)
validation_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(image_height, image_width),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        subset='validation')

class_list = [trace for trace in os.listdir(pathname) if not trace.startswith('.')]
# train_ints = [y.argmax() for y in train_generator]

# class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(train_ints),
#                                                  train_ints)

# print("class weights: ", class_weights)

num_epochs = 10 # used to be 2
early_stop_patience = 3
num_classes = len(class_list)


## Train the model
# clf.add(Conv2D(128, kernel_size=(2, 2), activation='relu', input_shape=(image_height, image_width, 3)))
# clf.add(MaxPooling2D(pool_size=(2, 2)))


# clf.add(Conv2D(64, kernel_size=(2, 2), activation='relu', kernel_initializer='he_uniform'))
# clf.add(MaxPooling2D(pool_size=(2, 2)))

# clf.add(Flatten())

# clf.add(Dense(64, activation='relu'))
clf.add(Dropout(0.3))
clf.add(Dense(num_classes, activation='softmax'))

clf.layers[0].trainable = False
clf.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# print(clf.layers[0].summary())
filepath="./checkpoints/mobile-weights.{epoch:02d}-{val_loss:.2f}.hdf5"

# stop training when a monitored quantity has stopped improving
early_stop = EarlyStopping(monitor = 'val_loss', patience = early_stop_patience)
# set of functions to be applied at given stages of training procedure
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='auto')
callbacks_list = [checkpoint, early_stop]

## Fit the Model
history = clf.fit_generator(train_generator, epochs=num_epochs, 
                                       steps_per_epoch=train_generator.samples // batch_size, 
                                       validation_data=validation_generator,
                                       validation_steps=validation_generator.samples // batch_size,
                                       shuffle=True, 
                                       )

# Plot the training and validation loss + accuracy
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')

plot_training(history)

## Prediction for Test Dataset
# data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
# test_generator = data_generator.flow_from_directory(directory=test_dir,
#                                                     target_size = (image_height, image_width),
#                                                     batch_size = 1,
#                                                     class_mode = None,
#                                                     shuffle = False,
# #                                                     seed = 123)

# test_generator.reset()
pred = clf.predict_generator(validation_generator, steps=len(validation_generator), verbose = 1)
predicted_class_indices=np.argmax(pred, axis = 1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
# print("labels ", labels)
# print("predictions ", predictions)

valid_labels = [labels[k] for k in validation_generator.labels]
print(valid_labels)


# test_files_names = test_generator.filenames
class_report = classification_report(valid_labels, predictions)
print(class_report)



