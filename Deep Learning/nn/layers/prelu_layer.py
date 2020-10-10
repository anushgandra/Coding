import numpy as np
from numba import njit, prange

from nn import Parameter
from .layer import Layer


class PReLULayer(Layer):
    def __init__(self, size: int, initial_slope: float = 0.1, parent=None):
        super(PReLULayer, self).__init__(parent)
        self.slope = Parameter(np.full(size, initial_slope))
        self.data = None
        self.size = size

    def forward(self, data):
        
        out1 = np.minimum(0,data)
        out1 = np.moveaxis(out1,1,-1)*self.slope.data
        out1 = np.moveaxis(out1,1,-1)
                     
        out2 = np.maximum(0,data)
        out = out1+out2

        self.data = data
        #print(self.slope.data)
        return out
    

    def backward(self, previous_partial_gradient):
        out1 = self.data>0
        out2 = np.moveaxis((self.data<=0),1,-1)*(self.slope.data)
        out2 = np.moveaxis(out2,1,-1)

        output = np.multiply((out1+out2),previous_partial_gradient)

        grad_mult = np.copy(self.data)
        grad_mult[grad_mult>0] = 0

        grad = np.multiply(previous_partial_gradient,grad_mult)

        if (self.size<2):
            self.slope.grad = np.sum(grad)
        else:
            grad_axes = np.arange(len(np.shape(grad)))
            sum_axes = grad_axes[0:-1]
            sum_axes = tuple(sum_axes)
            grad = np.moveaxis(grad,1,-1)
            self.slope.grad = (np.sum(grad,axis=sum_axes))
            
        return output
