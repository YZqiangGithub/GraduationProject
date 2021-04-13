from tensorflow import keras
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

subtrainLabel = pd.read_csv('./subtrainLabels.csv')
subtrain_feature = pd.read_csv('data/train/asm_imgfeature.csv')
subtrain = pd.merge(subtrainLabel,subtrain_feature,on = 'Id')
img_height, img_width = 40, 256

dataset = subtrain.values
lable = dataset[:,1] - 1
X = dataset[:,2:]
tmp = np.array(X)
X = tmp.reshape(len(tmp), img_height, img_width, 1)
X = X / 255.0

# datapath = './data/trainData/'
# train_image = []
# for i in tqdm(range(subtrainLabel.shape[0])):
#     txt = np.loadtxt(datapath + subtrainLabel['Id'][i])
#     txt = txt.astype('uint8')
#     # img = txt.reshape(-1, 1)
#     img = txt.resize(40,256)
#     img = img / 255
#     train_image.append(img)
# X = np.array(train_image)
#
# y = subtrainLabel['Class'].values - 1
# y = to_categorical(y)

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
