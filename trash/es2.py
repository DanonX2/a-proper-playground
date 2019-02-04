import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



data,labels = [np.array([]),np.random.random((1000,64))]

model=tf.keras.Sequential([
    layers.Dense(64, activation='relu')
    ])

model.compile(optimizer=tf.train.GradientDescentOptimizer(1),loss='mse',metrics=['mae'])

model.fit(data,labels,epochs=10,batch_size=32)