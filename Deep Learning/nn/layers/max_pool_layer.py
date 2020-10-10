import numbers

import numpy as np
from numba import njit, prange

from .layer import Layer


class MaxPoolLayer(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2, parent=None):
        super(MaxPoolLayer, self).__init__(parent)
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.padded_data = None
        self.max_ind = None
        

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data,stride,kernel_size,out,max_ind):
        [n,d,h,w] = data.shape
        
        jlim = (h-kernel_size) + 1 
        ilim = (w-kernel_size) + 1 
                  
        for l in prange(n):
            for o in prange(d):
                for j in range(0,jlim,stride):
                    for i in range(0,ilim,stride):
                        temp = data[l,o,j:j+(kernel_size),i:i+(kernel_size)]
                        max_val = np.max(temp)
                        out[l,o,j//stride,i//stride] = max_val
                        max_ind[l,o,j//stride,i//stride] = np.argmax(temp);
        return out

    def forward(self, data):
        padded = np.pad(data,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant')
        output_array = np.zeros((data.shape[0],data.shape[1],(padded.shape[2]-self.kernel_size)//self.stride + 1,(padded.shape[3]-self.kernel_size)//self.stride + 1),dtype=np.float32)

        self.max_ind = np.zeros_like(output_array,dtype=int)
        output = self.forward_numba(padded, self.stride, self.kernel_size,output_array,self.max_ind)

        self.padded_data = padded
        
        return output
        
    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, output, stride, k, padding, padded_data, max_ind):

        [n,d,h,w] = padded_data.shape
        [n,c,hout,wout] = previous_grad.shape
        
                      
        for l in prange(n):
            for v in prange(d):
                for j in range(hout):
                    for i in range(wout):
                        h1 = j*stride
                        h2 = h1+k
                        w1 = i*stride
                        w2 = w1+k
                        temp = np.zeros((k*k));
                        temp[max_ind[l,v,j,i]] = 1
                        temp2 = np.reshape(temp,(k,k))
                        output[l,v,h1:h2,w1:w2] += previous_grad[l,v,j,i]*temp2
            
         
        if(padding == 0):
            return(output[:,:,:,:])
        else:
            return(output[:,:,padding:-padding,padding:-padding])

    def backward(self, previous_partial_gradient):
        output_array = np.zeros((self.padded_data.shape),dtype = np.float32)
        output = self.backward_numba(previous_partial_gradient,output_array,self.stride,self.kernel_size,self.padding,self.padded_data,self.max_ind)
        
        return output

    def selfstr(self):
        return str("kernel: " + str(self.kernel_size) + " stride: " + str(self.stride))
