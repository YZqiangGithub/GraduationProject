#encoding:utf-8
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, Dense, Activation
from sklearn import feature_extraction

# Embedding
maxlen = 5000
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 75

subtrainLabel = pd.read_csv('./subtrainLabels.csv')
subtrain_feature = pd.read_csv('./op_seq.csv')
subtrain = pd.merge(subtrainLabel,subtrain_feature,on = 'Id')

d_train, d_test = train_test_split(subtrain, test_size = 0.4)
y_train = d_train['Class'].values - 1
y_test =  d_test['Class'].values - 1

vectorizer = feature_extraction.text.TfidfVectorizer(max_features=4000, ngram_range=(3,3))

ops_train = d_train['ops'].apply(lambda x: np.str_(x))
ops_test  = d_test['ops'].apply(lambda x: np.str_(x))
vectorizer.fit(ops_train)
x_train = vectorizer.transform(ops_train).toarray()
vectorizer.fit(ops_test)
x_test =  vectorizer.transform(ops_test).toarray()


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
model.add(Embedding(4000, embedding_size, input_length=maxlen))
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