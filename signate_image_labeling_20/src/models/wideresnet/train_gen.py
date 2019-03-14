"""
Train model with fit_generator, ImageDataGenerator
"""

import os 
import numpy as np
import pandas as pd
from skimage import io
import keras
import tensorflow as tf
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K
from model import get_model
from keras.preprocessing.image import ImageDataGenerator
import random as rn
from glob import glob

"""
fix seed
This ensures model reproducibility.
"""
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

train_path='../../../data/gen/train/'
validation_path='../../../data/gen/validation/'

the_number_of_images=50000
train_ratio=0.9
train_sample=int(the_number_of_images*train_ratio)
validation_sample=the_number_of_images-train_sample

batch_size=256
train_num_batches_per_epoch=int(the_number_of_images/batch_size)+1
validation_num_batches_per_epoch=int(validation_sample/batch_size)+1

"""
data augumentation
Using ImageDataGenerator, it is easy to generate some processed images.
"""
train_data_generator=ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.,
    zoom_range=0.3,
    channel_shift_range=0.1,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=1./255
    )
validation_data_generator=train_data_generator=ImageDataGenerator(rescale=1./255)

#data loading
train_generator=train_data_generator.flow_from_directory(
    directory=train_path,
    target_size=(32,32),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)

validation_generator=validation_data_generator.flow_from_directory(
    directory=validation_path,
    target_size=(32,32),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)

#model loading
model=get_model(input_shape=(32,32,3),n_class=20)

#comile
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

"""
Call back definition
ModelCheckpoint saves (model and weights) per epoch.
Further, if 'save_best_only=True', 'save_weights_only=True' and 'monitor='val_loss'', it saves only weights and only when validation loss developed.
"""
os.makedirs('weights/',exist_ok=True)
mc=keras.callbacks.ModelCheckpoint(os.path.join('weights/','epoch_{epoch:02d}-loss_{val_loss:.2f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

"""
In this code, we train model by 2 step training.
First, train with Adam(lr=0.0001), 15 epochs.
Second, train with Adam(lr=0.00001), 10 epochs.
"""
first_epoch=15
second_epoch=10

#step 1
history=model.fit_generator(generator=train_generator,
                      steps_per_epoch=train_num_batches_per_epoch,
                      epochs=first_epoch,
                      verbose=1,
                      callbacks=[mc],
                      validation_data=validation_generator,
                      validation_steps=validation_num_batches_per_epoch,
                      class_weight=None,
                      #max_queue_size=10,
                      workers=1,
                      use_multiprocessing=True,
                      shuffle=True,
                      initial_epoch=0)

#redefine Adam
opt = Adam(lr=0.00001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#load best weights in step 1
weight_path=sorted(glob('weights/*.hdf5'))[-1]
model.load_weights(weight_path)

#step 2
history=model.fit_generator(generator=train_generator,
                      steps_per_epoch=train_num_batches_per_epoch,
                      epochs=first_epoch+second_epoch,
                      verbose=1,
                      callbacks=[mc],
                      validation_data=validation_generator,
                      validation_steps=validation_num_batches_per_epoch,
                      class_weight=None,
                      #max_queue_size=10,
                      workers=1,
                      use_multiprocessing=True,
                      shuffle=True,
                      initial_epoch=first_epoch)