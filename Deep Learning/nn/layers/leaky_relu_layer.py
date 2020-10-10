from numba import njit, prange

from .layer import Layer


class LeakyReLULayer(Layer):
    def __init__(self, slope: float = -0.1, parent=None):
        super(LeakyReLULayer, self).__init__(parent)
        self.slope = slope
        self.data = None

    def forward(self, data):
        self.data = data
        data[data<=0] = self.slope*data[data<=0]
        return data

    def backward(self, previous_partial_gradient):
        self.data[self.data>0] = 1
        self.data[self.data<=0] = self.slope
        output = self.data*previous_partial_gradient
        return output
