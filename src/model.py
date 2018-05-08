from keras.preprocessing.text import Tokenizer
from input import DataReader


dr = DataReader('../data/sts-dev.csv')
texts = [t for t in dr.read_text()]
tk = Tokenizer()
tk.fit_on_texts(texts)
m = tk.texts_to_matrix(texts, mode='tfidf')
