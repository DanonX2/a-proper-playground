import tensorflow as tf
import numpy as np
import pandas as pd
import os
import math
from sklearn import metrics
from tensorflow.python.data import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

y = input('y:')
x = input('x:')
b = input('b:')

my_features = pd.DataFrame()
my_features['x'] = pd.Series([float(x)])
my_features['b'] = pd.Series([float(b)])

targets = pd.DataFrame()
targets['target'] = pd.Series([float(y)])

def my_input_fn():
    features = {key:np.array(value) for key,value in dict(my_features).items()}
    ds = Dataset.from_tensor_slices((features,targets['target'])).batch(1)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_model(
    learning_rate, 
    steps):
    
    feature_columns = set([
        tf.feature_column.numeric_column('x'),
        tf.feature_column.numeric_column('b')
        ])

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )
    for i in range(steps):
        linear_regressor.train(
            input_fn=lambda:my_input_fn(),
            steps=1
        )
        predictions = linear_regressor.predict(input_fn=lambda:my_input_fn())
        predictions = np.array([item['predictions'][0] for item in predictions])
        error = targets['target'] - predictions

        print("step: ", i+1, '   ',error)
    print('x: ',predictions)

train_model(0.05,10)