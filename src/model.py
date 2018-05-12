import datetime
import utils

import os.path as path
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import TensorBoard

from datareader import DataReader


def read_input_data(file_name):
    dr = DataReader(file_name)
    texts, scores = dr.read_data()
    tk = Tokenizer()
    tk.fit_on_texts(texts)
    x = tk.texts_to_matrix(texts, mode='tfidf')
    x = utils.matrix_to_input(x)
    y = utils.scores_to_categorical(scores)
    return x, y


x_train, y_train = read_input_data('../data/sts-train.csv')

num_rows, num_columns = x_train.shape
model = Sequential()
model.add(Dense(utils.NUM_CATEGORIES, input_shape=(num_columns,)))
model.add(Activation('linear'))
model.summary()

current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
logdir = path.join('logs', current_time)
tensorboardDisplay = TensorBoard(log_dir=logdir,
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=True,
                                 write_grads=True)

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae', 'acc'])
model.fit(x_train, y_train, epochs=500,
          callbacks=[tensorboardDisplay])

x_test, y_test = read_input_data('../data/sts-test.csv')
print(model.evaluate(x_test, y_test))
