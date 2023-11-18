from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

import MyLinearRegression
from MyGradientLinearRegression import MyGradientLinearRegression
from MySGDLinearRegression import MySGDLinearRegression
# from MyLogisticRegression import MyLogisticRegression
from MyRidgeRegression import MyRidgeRegression
from MySGDRidge import MySGDRidge
from MySGDLasso import MySGDLasso

import matplotlib.pyplot as plt

#ЛИНЕЙНЫЕ МОДЕЛИ ДЛЯ ЗАДАЧ РЕГРЕССИИ

def func_y(x):
    return 5 * x + 6

# по признакам сгенерируем значения таргетов с некоторым шумом
objects_num = 50
X = np.linspace(-5, 5, objects_num)
y = func_y(X) + np.random.randn(objects_num) * 5

# выделим половину объектов на тест
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)


print("1 My Linear Regression")
reg1 = MyLinearRegression.MyLinearRegression()
reg1.fit(X_train[:, np.newaxis], y_train)
preds1 = reg1.predict(X_test[:, np.newaxis])
w1 = reg1.get_weights()
mse1 = mean_squared_error(y_test, preds1)
print("weights:", w1)
print('Test MSE: ', mse1)
print("")

print("2 Sklearn Linear Regression")
reg2 = LinearRegression().fit(X_train[:, np.newaxis], y_train)
preds2 = reg2.predict(X_test[:, np.newaxis])
w2 = np.append(reg2.coef_, reg2.intercept_)
mse2 = mean_squared_error(y_test, preds2)
print("weights:", w2)
# print('Test MSE: ', mse2)


print("3 My Gradient Linear Regression")
reg3 = MyGradientLinearRegression()
loss3 = reg3.fit(X_train[:, np.newaxis], y_train, max_iter=100).get_losses()
preds3 = reg3.predict(X_test[:, np.newaxis])
w3 = reg3.get_weights()
mse3 = mean_squared_error(y_test, preds3)
print("weights:", w3)
print('Test MSE: ', mse3)
print("")

print("4 My SGD Linear Regression")
reg4 = MySGDLinearRegression(fit_intercept=True, n_sample = 4)
loss4 = reg4.fit(X_train[:, np.newaxis], y_train, max_iter=100).get_losses()
preds4 = reg4.predict(X_test[:, np.newaxis])
w4 = reg4.get_weights()
mse4 = mean_squared_error(y_test, preds4)
print("weights:", w4)
print('Test MSE: ', mse4)
print("")

print("5 MyRidgeRegression")
alpha = 1.0
reg5 = MyRidgeRegression(alpha=alpha)
reg5.fit(X_train[:, np.newaxis], y_train)
preds5 = reg5.predict(X_test[:, np.newaxis])
w5 = reg5.get_weights()
mse5 = mean_squared_error(y_test, preds5)
print("weights:", w5)
print('Test MSE: ', mse5)
print("")

print("6 MySGDRidgeRegression")
alpha = 1.0
reg6 = MySGDRidge(alpha=1, n_sample=20)
reg6.fit(X[:, np.newaxis], y, max_iter=1000, lr=0.01)
preds6 = reg6.predict(X_test[:, np.newaxis])
w6 = reg6.get_weights()
mse6 =  mean_squared_error(y_test, preds6)
print("weights:", w6)
print('Test MSE: ', mse6)
print("")


print("7 MySGDLasso")
reg7 = MySGDLasso(alpha=1, n_sample=4)
loss7 = reg7.fit(X[:, np.newaxis], y, max_iter=1000, lr=0.01).get_losses()
preds7 = reg7.predict(X_test[:, np.newaxis])
w7 = reg7.get_weights()
mse7 = mean_squared_error(y_test, preds7)
print("weights:", w7)
print('Test MSE: ', mse7)
print("")


print("8 Sklearn Ridge")
alpha = 1.0
reg8 = Ridge(alpha)
reg8.fit(X_train[:, np.newaxis], y_train)
preds8 = reg8.predict(X_test[:, np.newaxis])
w8 = np.append(reg8.coef_, reg8.intercept_)
mse8 = mean_squared_error(y_test, preds8)
print("weights:", w8)
print('Test MSE: ', mse8)
print("")

print("9 Sklearn Lasso")
alpha = 1.0
reg9 = Lasso(alpha)
reg9 = reg9.fit(np.hstack((X, X, X))[:, np.newaxis], np.hstack((y, y, y)))
preds9 = reg9.predict(X_test[:, np.newaxis])
w9 = np.append(reg9.coef_, reg9.intercept_)
mse9 = mean_squared_error(y_test, preds9)
print("weights:", w9)
print('Test MSE: ', mse9)
print("")


fig = plt.figure(figsize=(10, 7))

plt.title("Comparison linregs")

plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')
plt.plot(X_test, preds1, label=f'1 My Linear Regression {round(mse1,2)}')
plt.plot(X_test, preds2, label=f'2 Sklearn Linear Regression {round(mse2,2)}')
plt.plot(X_test, preds3, label=f'3 My Gradient Linear Regression {round(mse3,2)}')
plt.plot(X_test, preds4, label=f'4 My SGD Linear Regression {round(mse4,2)}')
plt.plot(X_test, preds5, label=f'5 MyRidgeRegression {round(mse5,2)}')
plt.plot(X_test, preds6, label=f'6 MySGDRidgeRegression {round(mse6,2)}')
plt.plot(X_test, preds7, label=f'7 MySGDLasso {round(mse7,2)}')
plt.plot(X_test, preds8, label=f'8 Sklearn Ridge {round(mse8,2)}')
plt.plot(X_test, preds9, label=f'9 Sklearn Lasso {round(mse9,2)}')

plt.grid(alpha=0.2)
plt.legend()
# plt.show()
fig.savefig("Comparison_linregs.png")


# #а это аналог из библиотеки Torch
# torch_linear_regression = nn.Linear(2,1)

# torch_loss_function = nn.BCEWithLogitsLoss()
# torch_optimizer = torch.optim.SGD(torch_linear_regression.parameters(), lr=0.05)

# #sigma = torch.sigmoid(logits)