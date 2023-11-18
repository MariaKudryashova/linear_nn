#простейшая нейронка или как вручную считать градиенты

import Funcs as F

plusGate = F.PlusGate()
multiGate = F.MultiplayGate()
multiConstGate = F.MultiplayConstantaGate()
expGate = F.ExponentaGate()
plusConstGate = F.PlusConstantaGate()
powMinOneGate = F.PowMinusOneXGate()
sigma = F.Sigmoid()
mse = F.MSE()


def iter(_is_pr, _x, _w0, _w1):
    #forward
    if (is_pr): print("forward")
    out = multiGate.forward(_x, _w1)
    if (is_pr): print("out: ", out)
    out = plusGate.forward(out, _w0)
    if (is_pr): print("out: ", out)
    out = sigma.forward(out)
    if (is_pr): print("sigma: ", out)

    y_pred = out
    
    loss = mse.forward(y_true, y_pred)
    if (is_pr): print("mse: ", loss)
    if (is_pr): print("backward")
    
    out = mse.backward(y_true, out)
    if (is_pr): print("mse", out)
    out = sigma.backward(out)
    if (is_pr): print("sigma", out)
    out = plusGate.backward(out)
    _grad_w0 = out[1]
    out = multiGate.backward(out[0])
    
    _grad_w1 = out[1]
    return _grad_w0, _grad_w1, loss, y_pred


x = 1
y_true = 1
w1 = 1
w0 = 1
w2 = 1
lr = 0.1

is_pr = False

for i in range(2000):
    grad_w0, grad_w1, loss, y_pred = iter(is_pr, x, w0, w1)
    if (is_pr): print(f"grad_w0: {grad_w0}, grad_w1: {grad_w1}")
    
    w0 -= lr * grad_w0
    w1 -= lr * grad_w1
           
    if (i % 500 == 0):
        print(f"iter:{i} \t mse:{loss} \t y_pred:{y_pred}")
