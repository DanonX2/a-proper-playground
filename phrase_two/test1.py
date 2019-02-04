import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#generate data
data_size = 10
feature = np.random.random((data_size,1)) * 100 
hidden_feature = np.random.random((data_size,1)) * 100 

target = feature*0.2 + hidden_feature

old_model = tf.keras.Sequential()
old_model.add(layers.Dense(64, activation='relu'))
old_model.add(layers.Dense(1, activation='linear'))

old_model.compile(
    optimizer=tf.train.AdamOptimizer(0.10),
    loss='mae',
    metrics=['mae'])

old_model.fit(feature,target,epochs=1000,batch_size=10)
example = np.random.random((200,1)) * 100 
predict = old_model.predict(example, batch_size=10)
plt.scatter(feature,target)
plt.scatter(example,predict,c='red',s=[1,1])
plt.show()
