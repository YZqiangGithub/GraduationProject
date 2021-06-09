import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF 
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

subtrainLabel = pd.read_csv('./subtrainLabels.csv')
subtrain_feature = pd.read_csv('data/asm_seq_train/asm_imgfeature.csv')
subtrain = pd.merge(subtrainLabel,subtrain_feature,on = 'Id')
lable = subtrain.Class

subtrain.drop(['Id','Class'], axis = 1, inplace = True)
x_train, x_test, y_train, y_test = train_test_split(subtrain,lable,test_size = 0.4)

srf = RF(n_estimators = 500, n_jobs = -1)
srf.fit(x_train,y_train)

y_pred = srf.predict(x_test)
print(accuracy_score(y_pred,y_test))
