from typing import Optional, Callable
import numpy as np

from numba import njit, prange

from nn import Parameter
from .layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(ConvLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.initialize()
        self.data = None
        self.padded_data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, weights, bias, stride, kernel_size,out):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        
        s = data.shape
        n = s[0]
        h = s[2]
        w = s[3]

        ws = weights.shape
        c = ws[1]

        jlim = (h-kernel_size) + 1 
        ilim = (w-kernel_size) + 1 

                  
        for l in prange(n):
            for o in prange(c):
                for j in range(0,jlim,stride):
                    for i in range(0,ilim,stride):
                        out[l,o,j//stride,i//stride] = np.sum((data[l,:,j:j+(kernel_size),i:i+(kernel_size)]*weights[:,o,:,:]))+bias[o]
                                                                        
        return out

    def forward(self, data):
        padded = np.pad(data,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant')
        output_array = np.zeros((data.shape[0],self.weight.data.shape[1],(padded.shape[2]-self.kernel_size)//self.stride + 1,(padded.shape[3]-self.kernel_size)//self.stride + 1),dtype=np.float32)

        output = self.forward_numba(padded, self.weight.data, self.bias.data, self.stride, self.kernel_size,output_array)
        
        self.data = data
        self.padded_data = padded
        return output

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(padded_data, prev_grad, weight, weight_grad, bias_grad,padded_output, padding, stride):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        
        [n,d,hin,win] =  padded_data.shape
        [n,c,hout,wout] = prev_grad.shape
        
        k = weight.shape[2]

                      
        for l in prange(n):
            for v in range(c):
                for j in range(hout):
                    for i in range(wout):
                        h1 = j*stride
                        h2 = h1+k
                        w1 = i*stride
                        w2 = w1+k
                        padded_output[l,:,h1:h2,w1:w2] += prev_grad[l,v,j,i]*weight[:,v,:,:]

        for v in prange(c):
            for l in range(n):
                for j in range(hout):
                    for i in range(wout):
                        h1 = j*stride
                        h2 = h1+k
                        w1 = i*stride
                        w2 = w1+k
                        temp = padded_data[l,:,h1:h2,w1:w2]*prev_grad[l,v,j,i]
                        weight_grad[:,v,:,:] += temp
                        bias_grad[v]+= prev_grad[l,v,j,i]
                        
                        
        if(padding == 0):
            return(padded_output[:,:,:,:])
        else:
            return(padded_output[:,:,padding:-padding,padding:-padding])
        
                        
    def backward(self, previous_partial_gradient):
        padded_output_array = np.zeros(self.padded_data.shape,dtype=np.float32)

        #output_array = np.zeros(self.data.shape,dtype=np.float32)

           
        back = self.backward_numba(self.padded_data, previous_partial_gradient, self.weight.data, self.weight.grad, self.bias.grad,padded_output_array, self.padding, self.stride)

        
        return back

    def selfstr(self):
        return "Kernel: (%s, %s) In Channels %s Out Channels %s Stride %s" % (
            self.weight.data.shape[2],
            self.weight.data.shape[3],
            self.weight.data.shape[0],
            self.weight.data.shape[1],
            self.stride,
        )

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(ConvLayer, self).initialize()
