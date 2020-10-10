import numpy as np

from .loss_layer import LossLayer


class SoftmaxCrossEntropyLossLayer(LossLayer):
    def __init__(self, reduction="mean", parent=None):
        """

        :param reduction: mean reduction indicates the results should be summed and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        self.reduction = reduction
        self.input_softmax = None
        self.targets = None
        super(SoftmaxCrossEntropyLossLayer, self).__init__(parent)

    def forward(self, logits, targets, axis=-1) -> float:
        """

        :param logits: ND non-softmaxed outputs. All dimensions (after removing the "axis" dimension) should have the same length as targets
        :param targets: (N-1)D class id integers.
        :param axis: Dimension over which to run the Softmax and compare labels.
        :return: single float of the loss.
        """
        #assuming logits is 2D:
        
        num_features = np.size(logits,axis)
        num_batches = targets.size
        one_hot_encoding = np.zeros((num_batches,num_features),dtype=np.float32)
        rows = np.arange(num_batches)
        one_hot_encoding[rows,targets] = 1.0

        self.targets = one_hot_encoding

        logits = logits - np.amax(logits,axis=axis,keepdims=True)
        
        log_logits = logits - np.log(np.sum(np.exp(logits),axis=axis,keepdims=True))

        self.input_softmax = np.exp(logits)/(np.sum(np.exp(logits),axis=axis,keepdims=True))
        
        H = -1*log_logits*one_hot_encoding
        temp2 = np.nonzero(H)
        H = H[temp2[0],temp2[1]]

        if(self.reduction == "sum"):
            H = np.sum(H)
        else:
            H = np.divide(np.sum(H),num_batches)
        return H


    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        y = (self.input_softmax - self.targets)
        if(self.reduction=="mean"):
            y = y/np.shape(self.targets)[0]
        return y
