import numpy as np
import MyLogisticSGDRegression
class MyElasticLogisticRegression(MyLogisticSGDRegression.MyLogisticSGDRegression):
    def __init__(self, l1_coef, l2_coef):
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.w = None

    
    
    def get_grad(self, X_batch, y_batch, predictions):
        """
        Принимает на вход X_batch с уже добавленной колонкой единиц. 
        Выдаёт градиент функции потерь в логистической регрессии с регуляризаторами
        как сумму градиентов функции потерь на всех объектах батча + регуляризационное слагаемое
        ВНИМАНИЕ! Нулевая координата вектора весов -- это BIAS, а не вес признака. 
        Bias в регуляризационные слагаемые не входит. Также не нужно ДЕЛИТЬ ГРАДИЕНТ НА РАЗМЕР БАТЧА:
        нас интересует не среднее, а сумма. 
  
        Выход -- вектор-столбец градиентов для каждого веса (np.array[n_features + 1])
        """


        grad_basic = np.dot(X_batch.T, (predictions - y_batch))
        #print("grad_basic", grad_basic)
        
        
        grad_l1 = self.l1_coef * np.sign(self.w)
        grad_l1[0] = 0
        #print(grad_l1)
        
        grad_l2 = 2 * self.l2_coef * self.w
        grad_l2[0] = 0
        #print("grad_l2", grad_l2)
        
        
        
        assert grad_l1[0] == grad_l2[0] == 0, "Bias в регуляризационные слагаемые не входит!"
        assert grad_basic.shape == grad_l1.shape == grad_l2.shape == (X_batch.shape[1],) , "Градиенты должны быть столбцом из k_features + 1 элементов"
        
        return grad_basic + grad_l1 + grad_l2