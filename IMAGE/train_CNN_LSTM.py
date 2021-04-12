from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, Dense, Activation
from tensorflow.keras.preprocessing import sequence

subtrainLabel = pd.read_csv('./subtrainLabels.csv')
subtrain_feature = pd.read_csv('data/train/asm_imgfeature.csv')
subtrain = pd.merge(subtrainLabel,subtrain_feature,on = 'Id')

#data process
total_len = 1600

# Embedding
vocablen = 400
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 100

dataset = subtrain.values
lable = dataset[:,1] - 1
X = dataset[:,2:]
tmp = np.array(X)
X = tmp.reshape(len(tmp), total_len,1)


x_train, x_test, y_train, y_test = train_test_split(X,lable,test_size = 0.3)

x_train = sequence.pad_sequences(x_train, maxlen=total_len)
x_test = sequence.pad_sequences(x_test, maxlen=total_len)
# x_train = keras.backend.cast_to_floatx(x_train)
# x_test = keras.backend.cast_to_floatx(x_test)
y_train = keras.backend.cast_to_floatx(y_train)
y_test = keras.backend.cast_to_floatx(y_test)

model = keras.Sequential()
model.add(Embedding(vocablen, embedding_size, input_length=total_len))
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