import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
model=tf.keras.Sequential([layers.Dense(64, activation='relu'),layers.Dense(64, activation='relu'),layers.Dense(10, activation='softmax')])
model.compile(optimizer=tf.train.GradientDescentOptimizer(1),loss='mse',metrics=['mae'])
data,labels = [np.random.random((1000,32)),np.random.random((1000,10))]
model.fit(data,labels,epochs=10,batch_size=32)