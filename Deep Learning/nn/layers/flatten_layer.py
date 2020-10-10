from .layer import Layer
import numpy as np

class FlattenLayer(Layer):
    def __init__(self, parent=None):
        super(FlattenLayer, self).__init__(parent)
        self.input_shape = None

    def forward(self, data):
        [n,d,h,w] = data.shape
        output = np.reshape(data,(n,d*h*w))
        self.input_shape = data.shape
        return output

    def backward(self, previous_partial_gradient):
        output = np.reshape(previous_partial_gradient,self.input_shape)
        return output
