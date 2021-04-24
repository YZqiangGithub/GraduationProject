from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, LSTM, Dense, Activation
from tensorflow.keras.preprocessing import sequence

subtrainLabel = pd.read_csv('./subtrainLabels.csv')
subtrain_feature = pd.read_csv('data/asm_seq_train/asm_imgfeature.csv')
subtrain = pd.merge(subtrainLabel,subtrain_feature,on = 'Id')
total_len = 1600
# Embedding
vocablen = 400
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

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
y_train = keras.backend.cast_to_floatx(y_train)
y_test = keras.backend.cast_to_floatx(y_test)

model = keras.Sequential()
model.add(Conv1D(filters,
                 kernel_size,
                 input_shape=(total_len, 1),
                 activation='relu'
                 ))
model.add(Conv1D(filters, kernel_size, activation='relu'))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Conv1D(filters, kernel_size, activation='relu'))
model.add(Conv1D(filters, kernel_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(128))
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