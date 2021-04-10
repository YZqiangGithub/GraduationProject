#encoding:utf-8

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, Dense, Activation

from sklearn.feature_extraction.text import TfidfVectorizer


# Embedding
embedding_size = 128

# Convolution
kernel_size = 3
filters = 128
pool_size = 3

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 100

subtrainLabel = pd.read_csv('./subtrainLabels.csv')
subtrain_feature = pd.read_csv('./op_seq.csv')
subtrain = pd.merge(subtrainLabel,subtrain_feature,on = 'Id')


train_df, test_df = train_test_split(subtrain, test_size = 0.4)
y_train = train_df['Class'].values - 1
y_test =  test_df['Class'].values - 1

tv = TfidfVectorizer(max_features=10000, ngram_range=(4,4))

train_df['ops'].fillna('__na__', inplace=True)
test_df['ops'].fillna('__na__', inplace=True)

ops_train = train_df['ops'].apply(lambda x: np.str_(x))
ops_test  = test_df['ops'].apply(lambda x: np.str_(x))
x_train = tv.fit_transform(ops_train).toarray()
x_test =  tv.transform(ops_test).toarray()

from sklearn.feature_selection import SelectKBest, chi2
ch2 = SelectKBest(chi2, k=4000)
x_train = ch2.fit_transform(x_train, y_train)
x_test = ch2.transform(x_test)

# voc_size = len(tv.vocabulary_) + 1
voc_size = 4001
# model = keras.Sequential()
# model.add(Embedding(4000, embedding_size, input_length=maxlen))
# model.add(Dropout(0.5))
# model.add(Conv1D(filters,
#                  kernel_size,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
# model.add(MaxPooling1D(pool_size=pool_size))
# model.add(LSTM(lstm_output_size,recurrent_dropout=0.5))
# model.add(Dense(9), ac)
#
# model.compile(optimizer='adam', loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

model = keras.Sequential()
model.add(Embedding(voc_size, embedding_size))
model.add(Dropout(0.5))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size, recurrent_dropout=0.5))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))
# model.add(Activation('softmax'))

model.compile(optimizer='adam', loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)