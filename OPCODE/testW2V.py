#encoding:utf-8

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, Dense, Activation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

def process_ops(total_ops):


subtrainLabel = pd.read_csv('./subtrainLabels.csv')
subtrain_feature = pd.read_csv('./op_seq.csv')
subtrain = pd.merge(subtrainLabel,subtrain_feature,on = 'Id')

train_df, test_df = train_test_split(subtrain, test_size = 0.4)
y_train = train_df['Class'].values - 1
y_test =  test_df['Class'].values - 1

train_ops = process_ops(train_df.ops)

tokenizer = Tokenizer()
tokenizer.fit_on_text(train_ops)

encoded_ops = tokenizer.texts_to_sequences(train_ops)

max_length = max([len(s.split()) for s in train_ops])
X_train = pad_sequences(encoded_ops, maxlen = max_length, padding ='post')

test_ops = process_ops(test_df.ops)
encoded_ops = tokenizer.texts_to_sequences(test_ops)

X_test = pad_sequences(encoded_ops, maxlen = max_length, padding='post')

vocab_size = len(tokenizer.word_index) + 1


