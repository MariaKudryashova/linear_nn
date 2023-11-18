import math

import numpy as np
from numpy.random import randn

#реализация блока с умножением
class MultiplayGate():                    
        
    def forward(self, x, y):
        z = x*y
        self.x = x
        self.y = y
        return z
    
    def backward(self, dz):
        # print("x: ", self.x, "y:", self.y, "dz: ", dz)
        grad_x = self.y * dz
        grad_y = self.x * dz
        return [grad_x, grad_y]

#реализация блока со сложением
class PlusGate():                    
        
    def forward(self, x, y):
        z = x + y
        return z
    
    def backward(self, dz):
        dx = dz
        dy = dz
        return [dx, dy]

#реализация блока экспоненты
class ExponentaGate():                    
        
    def forward(self, x):  
        self.grad_x = math.exp(x)
        return self.grad_x
    
    def backward(self, dz):
        return dz * self.grad_x

#реализация блока умножения на константу
class MultiplayConstantaGate():                    
        
    def forward(self, x, a): 
        return x * a
    
    def backward(self, dz, a):        
        return dz * a

#реализация блока сложения с константой
class PlusConstantaGate():                    
        
    def forward(self, x, a):
        self.a = a
        return x + a
    
    def backward(self, dz):        
        return dz * self.a

#реализация блока 1/x
class PowMinusOneXGate():                    
        
    def forward(self, x): 
        self.x = x
        return 1/x
    
    def backward(self, dz):  
        grad_x = -(1/((self.x)*(self.x)))
        return dz * grad_x

#функции активации
class Sigmoid():     

    def forward(self, x):    
        self.grad = 1/(1 + math.exp(-x))
        return self.grad
        
    def backward(self, input):
        # print("grad sigma", self.grad, input)
        a = self.grad * (1 - self.grad)         
        return a * input

#функции ошибки
class MSE():

    def forward(self, y, y_pred):
        self.grad = np.square(y_pred - y).sum() #(y_true - x)*(y_true - x)   
        return self.grad
    
    def backward(self, y_true, y_pred):
        out = 2 * (y_pred - y_true)
        return out

    
    