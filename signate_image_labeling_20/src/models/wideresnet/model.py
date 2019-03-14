"""
Model definition
Requirement:
keras_contrib(https://github.com/keras-team/keras-contrib)
"""

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras_contrib.applications import WideResidualNetwork

def get_model(input_shape,n_class):
    base_model=WideResidualNetwork(input_shape=input_shape)
    base_model.layers.pop()
    x=base_model.layers[-1].output
    prediction=Dense(n_class,activation='softmax')(x)
    model=Model(inputs=base_model.input,outputs=prediction)
    return model