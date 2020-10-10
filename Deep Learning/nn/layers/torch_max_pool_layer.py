import numbers

import torch
import torch.nn.functional as F

from tests import utils
from .layer import Layer


class TorchMaxPoolLayer(Layer):
    def __init__(self, kernel_size, stride=1, parent=None):
        super(TorchMaxPoolLayer, self).__init__(parent)
        if isinstance(kernel_size, numbers.Number):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (self.kernel_size[0] - 1) // 2
        self.data = None
        self.output = None
        self.maxpool_layer = torch.nn.MaxPool2d(self.kernel_size, self.stride)

    def forward(self, data):
        self.data = utils.from_numpy(data)
        self.data.requires_grad_(True)
        self.output = F.max_pool2d(self.data, self.kernel_size, self.stride, self.padding)
        return utils.to_numpy(self.output)

    def backward(self, previous_partial_gradient):
        gradients = utils.from_numpy(previous_partial_gradient)
        new_gradients = torch.autograd.grad(self.output, self.data, gradients, retain_graph=False)[0]
        return utils.to_numpy(new_gradients)

    def selfstr(self):
        return "Kernel: %s Stride %s" % (self.kernel_size, self.stride)
