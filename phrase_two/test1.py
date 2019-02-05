import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#generate data
data_size = 10
feature = np.random.random((data_size,1))
hidden_feature = np.random.random((data_size,1))

target = feature*0.2 + hidden_feature
#target = feature*feature

#hyperparameter
epochs = 5000
lr = 0.0001
decay_rate = lr*2 / epochs
momentum = 0.5
optimizer = tf.keras.optimizers.SGD(lr=lr,decay=decay_rate,momentum=momentum)


hidden_feature_predictor = tf.keras.Sequential([
    layers.Dense(50, activation='relu',kernel_initializer='random_uniform',bias_initializer='random_uniform'),
    layers.Dense(50, activation='relu',kernel_initializer='random_uniform',bias_initializer='random_uniform'),
    layers.Dense(50, activation='relu',kernel_initializer='random_uniform',bias_initializer='random_uniform'),
    layers.Dense(1, activation='linear',kernel_initializer='random_uniform',bias_initializer='random_uniform')
])

hidden_feature_predictor.compile(
    optimizer = optimizer,
    loss='mae',
    metrics=['mae']
)
hidden_feature_predictor.fit()










old_model = tf.keras.Sequential()
old_model.add(layers.Dense(50, activation='relu',kernel_initializer='random_uniform',bias_initializer='random_uniform'))
old_model.add(layers.Dense(1, activation='linear',kernel_initializer='random_uniform',bias_initializer='random_uniform'))


old_model.compile(
    optimizer=optimizer,
    loss='mae',
    metrics=['mae'])


old_model.fit(feature,target,epochs=epochs,batch_size=1)

example = np.random.random((200,1))
predict = old_model.predict(example)
plt.scatter(feature,target)
plt.scatter(example,predict,c='red',s=[1,1])
plt.show()
