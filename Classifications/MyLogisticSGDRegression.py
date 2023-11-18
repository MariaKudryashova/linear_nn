import numpy as np


def logit(x, w):
    return np.dot(x, w)

def sigmoid(h):
    return 1. / (1 + np.exp(-h))

def generate_batches(X, y, batch_size):
    """
    param X: np.array[n_objects, n_features] --- матрица объекты-признаки
    param y: np.array[n_objects] --- вектор целевых переменных
    """
    assert len(X) == len(y)
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))
    X2 = np.array(X) #[x for x in range(len(X))]
    y2 = np.array(y) #[x for x in range(len(y))]
    
    for i in range(0, len(X)):
        X2[i] = X[perm[i]]
        y2[i] = y[perm[i]]
        
    n = round(len(X)/batch_size)
    
    for i in range(n):  
        
        #print(perm_x[i*batch_size:batch_size*(i+1)], perm_y[i*batch_size:batch_size*(i+1)])
        i0 = i*batch_size
        i1 = batch_size*(i+1)
        yield X2[i0:i1], y2[i0:i1]


class MyLogisticSGDRegression(object):

    def __init__(self):
        self.w = None

    def fit(self, X_train, y, epochs=5, lr=0.1, batch_size=5):
        n, k = X_train.shape        
        X_train = np.concatenate((np.ones((n, 1)), X_train), axis=1)  
        if self.w is None:
            np.random.seed(42)
            self.w = np.random.randn(k + 1)
        
        losses = []

        for i in range(epochs):
            for X_batch, y_batch in generate_batches(X_train, y, batch_size):
                #В X_train уже добавлен вектор 1
                predictions = self._predict_proba_internal(X_batch)
                loss = self.__loss(predictions, y_batch)

                #assert (np.array(loss).shape == tuple()), "Лосс должен быть скаляром!" 

                losses.append(loss)

                grad = self.get_grad(X_batch, y_batch, predictions)
                self.w -= lr * grad
        #print("loses",losses)
        return losses

    def get_grad(self, X_batch, y_batch, predictions):       

        grad_basic = np.dot(X_batch.T, (predictions - y_batch)) 
        #assert grad_basic.shape == (X_batch.shape[1],) , "Градиенты должны быть столбцом из k_features + 1 элементов"

        return grad_basic

    def predict_proba(self, X):
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logit(X_, self.w))

    def _predict_proba_internal(self, X): 
        """
        Возможно, вы захотите использовать эту функцию вместо predict_proba, поскольку
        predict_proba конкатенирует вход с вектором из единиц, что не всегда удобно
        для внутренней логики вашей программы
        """
        
        return sigmoid(logit(X, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w
        # copy тут используется неспроста. Если copy не использовать, то get_weights()
        # выдаст ссылку на объект, а, значит, модифицируя результат применения функции
        # get_weights(), вы модифицируете и веса self.w. Если вы хотите модифицировать веса, 
        # (например, в fit), используйте self.w

    def __loss(self, y, p):  
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    