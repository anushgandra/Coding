import numpy as np
from numba import njit, prange

from .layer import Layer


class ReLULayer(Layer):
    def __init__(self, parent=None):
        super(ReLULayer, self).__init__(parent)
        self.data = None

    def forward(self, data):
        y = np.maximum(0,data)
        self.data = data
        return y

    def backward(self, previous_partial_gradient):
        y = np.multiply(previous_partial_gradient,(self.data>0))
        return y


class ReLUNumbaLayer(Layer):
    def __init__(self, parent=None):
        super(ReLUNumbaLayer, self).__init__(parent)
        self.data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        data_length = data.shape
        data_length = data_length[0]
        for i in range(0,data_length):
            if(data[i] <= 0):
                data[i] = 0
            else:
                continue
                   
        return data

    def forward(self, data):
        self.data = data
        input_shape = np.shape(data)
        flattened_inp = data.flatten()
        flattened_output = self.forward_numba(flattened_inp)
        output = np.reshape(flattened_output,input_shape)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        grad_shape = grad.shape
        grad_length = grad_shape[0]
        for i in range(0,grad_length):
            if(data[i]<=0):
                grad[i]=0
            else:
                continue
        return(grad)

        
    def backward(self, previous_partial_gradient):
        prev_shape = np.shape(previous_partial_gradient)
        flat_data = self.data.flatten()
        flat_prev = previous_partial_gradient.flatten()
        flat_gradient =  self.backward_numba(flat_data, flat_prev)
        gradient = np.reshape(flat_gradient,prev_shape)
        return gradient
