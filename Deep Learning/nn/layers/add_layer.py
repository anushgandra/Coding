from collections.abc import Iterable
from typing import Tuple

from .layer import Layer

import numpy as np

class AddLayer(Layer):
    def __init__(self, parents):
        super(AddLayer, self).__init__(parents)
        self.data_shape = None

    def forward(self, inputs: Iterable):
        # TODO: Add all the items in inputs. Hint, python's sum() function may be of use.
        output = sum(inputs)
        self.data_shape = len(inputs)
        return output

    def backward(self, previous_partial_gradient) -> Tuple[np.ndarray, ...]:
        # TODO: You should return as many gradients as there were inputs.
        #   So for adding two tensors, you should return two gradient tensors corresponding to the
        #   order they were in the input.
        output = (previous_partial_gradient,)*self.data_shape
                        
        return output
