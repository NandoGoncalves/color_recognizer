
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

import pickle

df = pd.read_csv("./images/color_range.csv", sep=',')

print(df.head())


data = df.drop('color', axis =1)
target = df.color


print(data.shape, target.shape)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=123)

dt_clf = DecisionTreeClassifier(criterion = "entropy", max_depth=4, random_state= 123)

dt_clf.fit(X_train, y_train)

y_pred = dt_clf.predict(X_test)

print(pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))

'''
Classe prédite  Blue  Other
Classe réelle
Blue             600      0
Other             23    577
'''

filename = './models/model_03.sav'
pickle.dump(dt_clf, open(filename, 'wb'))


