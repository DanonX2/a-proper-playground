#load libraries
from __future__ import print_function
import tensorflow as tf
import math
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as pyplot
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.python.data import Dataset

#config setup
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#load data from cvs to pandas dataframe
cvs = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
california_housing_dataframe = pd.read_csv(cvs, sep=",")

#randomize the data + change unit
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0    

#define the input feature: total_rooms
my_feature = california_housing_dataframe[["total_rooms"]]
#configure a numeric feature column for total rooms
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

#define the label
targets = california_housing_dataframe["median_house_value"]