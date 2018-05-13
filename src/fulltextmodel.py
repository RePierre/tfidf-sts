import datetime
import utils
import os.path as path
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import TensorBoard

from datareader import DataReader


def read_dataset(dataset_file):
    dr = DataReader(dataset_file)
    return dr.read_data()


def read_input_data(train_dataset, test_dataset):
    texts_train, scores_train = read_dataset(train_dataset)
    texts_test, scores_test = read_dataset(test_dataset)
    tk = Tokenizer()
    texts = texts_train + texts_test
    tk.fit_on_texts(texts)
    x = tk.texts_to_matrix(texts, mode='tfidf')
    x_train, x_test = x[:len(texts_train)], x[len(texts_train):]
    x_train, x_test = utils.matrix_to_input(x_train), utils.matrix_to_input(x_test)
    y_train, y_test = utils.scores_to_categorical(scores_train), utils.scores_to_categorical(scores_test)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = read_input_data('../data/sts-train.csv', '../data/sts-test.csv')

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
print(model.evaluate(x_test, y_test))
