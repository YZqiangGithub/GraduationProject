from tensorflow import keras
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, LSTM, Dense, Activation
from tensorflow.keras.preprocessing import sequence

datapath = './data/trainData/'
train = pd.read_csv('subtrainLabels.csv')
total_len = 10000
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

train_image = []
for i in tqdm(range(train.shape[0])):
    txt = np.loadtxt(datapath + train['Id'][i])
    txt = txt.astype('uint8')
    img = txt.reshape(-1, 1)
    img = img / 255
    train_image.append(img)
X = np.array(train_image)

y = train['Class'].values - 1
y = to_categorical(y)

x_train, x_test,y_train, y_test = train_test_split(X, y , test_size=0.3)

x_train = sequence.pad_sequences(x_train, maxlen=total_len)
x_test = sequence.pad_sequences(x_test, maxlen=total_len)

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

model.compile(optimizer='adam', loss= 'categorical_crossentropy',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)