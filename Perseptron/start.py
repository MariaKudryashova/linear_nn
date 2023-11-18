#НС через граф и далее, если сделать матрицами / тензорами по такой же логике,
#как в библиотеках питона

import Funcs as F
import ComputationalGraph as cg


x = 1
y_true = 1
w1 = 1
w0 = 1
lr = 0.1

is_pr = False


for i in range(2000):

    g = cg.ComputationalGraph(y_true, is_pr)
    loss, y_pred = g.forward(x, w0, w1)

    if (i % 500 == 0):
        print("loss", loss, "y_pred", y_pred)

    grads = g.backward()
    # print("grads", grads)

    w0 -= lr*grads[0]
    w1 -= lr*grads[1]


