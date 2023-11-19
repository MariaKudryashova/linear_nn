import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs

from MyLogisticRegression import MyLogisticRegression
from MyLogisticSGDRegression import MyLogisticSGDRegression
from MyLogisticRegression import MyLogisticRegression
from MyElasticLogisticRegression import MyElasticLogisticRegression

X, y = make_blobs(n_samples=1000, centers=[[-2,0.5],[2,-0.5]], cluster_std=1, random_state=42)

# выделим половину объектов на тест
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

#ЛИНЕЙНЫЕ МОДЕЛИ ДЛЯ ЗАДАЧ КЛАССИФИКАЦИИ

print("1 My Logistic Regression")
reg1 = MyLogisticRegression()
reg1.fit(X_train, y_train, max_iter = 1000)
preds1 = reg1.predict_proba(X_test)
w1 = reg1.get_weights()
mse1 = mean_squared_error(y_test, preds1)
print("weights:", w1)
print('Test MSE: ', mse1)
print("")

print("2 My Logistic SGD Regression")
reg2 = MyLogisticSGDRegression()
reg2.fit(X_train, y_train)
preds2 = reg2.predict_proba(X_test)
w2 = reg2.get_weights()
mse2 = mean_squared_error(y_test, preds2)
print("weights:", w2)
print('Test MSE: ', mse2)
print("")

print("3 My Elastic Logistic Regression")
reg3 = MyElasticLogisticRegression(.2, .2)
reg3.fit(X_train, y_train, epochs=1000)
preds3 = reg3.predict_proba(X_test)
w3 = reg3.get_weights()
mse3 = mean_squared_error(y_test, preds3)
print("weights:", w3)
print('Test MSE: ', mse3)
print("")

print("4 Sklearn Logistic Regression")
reg4 = LogisticRegression()
reg4.fit(X_train, y_train)
preds4 = reg4.predict(X_test)
w4 = np.append(reg4.coef_, reg4.intercept_)
mse4 = mean_squared_error(y_test, preds4)
print("weights:", w4)
print('Test MSE: ', mse4)
print("")

print("5 KNeighborsClassifier")
reg5 = KNeighborsClassifier(n_neighbors=3, p=2)
reg5.fit(X_train, y_train)
preds5 = reg5.predict(X_test)
mse5 = mean_squared_error(y_test, preds5)
print('Test MSE: ', mse5)
print("")


values = [0, mse1, mse2, mse3, mse4, mse5]

labels = ['Real',
'1',
'2',
'3',
'4',
'5']


def autolabel(rects, labels=None, height_factor=1.01):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if labels is not None:
            try:
                label = labels[i]
            except (TypeError, KeyError):
                label = ' '
        else:
            label = '%d' % int(height)
        print(rect.get_x())
        ax.text(rect.get_x() + rect.get_width()/2., height_factor*height,
                '{}'.format(label),
                ha='center', va='bottom')



fig = plt.figure(figsize=(10, 7))

plt.title("Comparison classes")

bars = plt.bar(labels, values)
ax = plt.gca()
autolabel(ax.patches, height_factor = 1.01)
# plt.show()
fig.savefig("Comparison_classes.png")