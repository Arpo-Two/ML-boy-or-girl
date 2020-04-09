import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('wyr.csv')

predict = 'gender'

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
model = KNeighborsClassifier(n_neighbors=815)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
predicted = model.predict(x_test)
for x in range(len(predicted)):
    print('Predicted: ', predicted[x], 'Data: ', x_test[x], 'Actual: ', y_test[x])
print(acc)
