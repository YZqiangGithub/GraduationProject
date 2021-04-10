from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

subtrainLabel = pd.read_csv('./subtrainLabels.csv')
subtrain_feature = pd.read_csv('./data/train/imgfeature.csv')
subtrain = pd.merge(subtrainLabel,subtrain_feature,on = 'Id')
img_height, img_width = 40, 40

dataset = subtrain.values
lable = dataset[:,1] - 1
X = dataset[:,2:]
tmp = np.array(X)
X = tmp.reshape(len(tmp), img_height, img_width, 1)
X = X / 255.0

x_train, x_test, y_train, y_test = train_test_split(X,lable,test_size = 0.4)

x_train = keras.backend.cast_to_floatx(x_train)
x_test = keras.backend.cast_to_floatx(x_test)
y_train = keras.backend.cast_to_floatx(y_train)
y_test = keras.backend.cast_to_floatx(y_test)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(40, 40, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(9))

model.compile(optimizer='adam', loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=75)

test_loss, test_accu = model.evaluate(x_test,y_test, verbose=2)

print("accuracy : ", test_accu)
