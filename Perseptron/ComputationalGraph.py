#реализация графа
import Funcs as F

class ComputationalGraph():    

    def __init__(self, y, is_pr):
        super(ComputationalGraph, self).__init__()
        self.is_pr = is_pr
        self.nodes = [            
            F.MultiplayGate(),
            F.PlusGate(),
            F.Sigmoid(),
            F.MSE()
        ]

        self.nodes_reversed = list(self.nodes)
        self.nodes_reversed.reverse()
        self.y_true = y
   

    def forward(self, _x, _w0, _w1):
        #for gate in self.nodes():
        out = self.nodes[0].forward(_x, _w1)
        if (self.is_pr): print("1", out)
        out = self.nodes[1].forward(out, _w0)
        if (self.is_pr): print("2", out)
        self.y_pred = self.nodes[2].forward(out)
        if (self.is_pr): print("sigma", self.y_pred)
        self.loss = self.nodes[3].forward(self.y_true, self.y_pred)
        if (self.is_pr): print("loss", self.loss)   

        return  self.loss, self.y_pred



    def backward(self):
        if (self.is_pr): print("backward")
        #for gate in reversed(self.nodes()):
        out = self.nodes_reversed[0].backward(self.y_true, self.y_pred) 
        if (self.is_pr): print("mse", out)

        out = self.nodes_reversed[1].backward(out)
        if (self.is_pr): print("sigma", out)

        self.grads = self.nodes_reversed[2].backward(out) 
        
        return self.grads