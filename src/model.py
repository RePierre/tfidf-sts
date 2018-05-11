import math
import datetime

import os.path as path
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense

from datareader import DataReader

NUM_CATEGORIES = 6


def matrix_to_input(matrix):
    num_rows, num_columns = matrix.shape
    result = []
    for i in range(int(num_rows / 2)):
        t1 = matrix[2 * i]
        t2 = matrix[2 * i + 1]
        c = np.concatenate((t1, t2), axis=0)
        result.append(c)
    result = np.asarray(result)
    return result


def scores_to_categorical(scores):
    result = []
    for score in scores:
        a = np.zeros(NUM_CATEGORIES)
        fractional, integer = math.modf(score)
        integer = int(integer)
        if fractional == 0.:
            a[integer] = 1.
        else:
            a[integer + 1] = fractional
        result.append(a)
    result = np.asarray(result)
    return result


dr = DataReader('../data/sts-dev.csv')
texts, scores = dr.read_data()
tk = Tokenizer()
tk.fit_on_texts(texts)
x_train = tk.texts_to_matrix(texts, mode='tfidf')
x_train = matrix_to_input(x_train)
y_train = scores_to_categorical(scores)

num_rows, num_columns = x_train.shape
model = Sequential()
model.add(Dense(NUM_CATEGORIES, input_shape=(num_columns,)))
model.add(Activation('linear'))
model.summary()

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae', 'acc'])
model.fit(x_train, y_train, epochs=50)
