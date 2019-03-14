"""
Predict labels of test data

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

#fix seed
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

test_path='../../../data/test/'

test_sample=10000
batch_size=256
test_num_batches_per_epoch=int(test_sample/batch_size)+1

#data augumentation
test_data_generator=ImageDataGenerator(rescale=1./255)

"""
data loading
Don't forget to set 'shuffle = False' when predicting.
"""
test_generator=test_data_generator.flow_from_directory(
    directory=test_path,
    target_size=(32,32),
    batch_size=batch_size,
    color_mode="rgb",
    shuffle = False,
    class_mode='categorical'
)

#model loading
model=get_model(input_shape=(32,32,3),n_class=20)

#load best weight
weight_path=sorted(glob('weights/*.hdf5'))[-1]
print(weight_path)
model.load_weights(weight_path)

#predict
predicted=model.predict_generator(generator=test_generator,
                      steps=test_num_batches_per_epoch,
                      verbose=1,
                      workers=1,
                      use_multiprocessing=True,
                      )

#save predicted labels
filenames=[os.path.basename(path) for path in test_generator.filenames]
submit=pd.concat([pd.DataFrame(filenames),pd.DataFrame(predicted)],axis=1)
submit.to_csv('submission.csv',index=False,header=False)