from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

subtrainLabel = pd.read_csv('subtrainLabels.csv')
subtrain_feature = pd.read_csv("./4gramfeature.csv")
subtrain = pd.merge(subtrainLabel,subtrain_feature, on = 'Id')
label = subtrain.Class
subtrain.drop(["Class", "Id"], axis = 1, inplace = True)
subtrain = subtrain.values

x_trian, x_test, y_trian, y_test = train_test_split(subtrain, label, test_size = 0.4)

srf = RF(n_estimators = 500, n_jobs = -1)
srf.fit(x_trian,y_trian)


y_pred = srf.predict(x_test)
print(accuracy_score(y_pred,y_test))