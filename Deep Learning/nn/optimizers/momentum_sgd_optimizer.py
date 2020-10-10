from .base_optimizer import BaseOptimizer


class MomentumSGDOptimizer(BaseOptimizer):
    def __init__(self, parameters, learning_rate, momentum=0.9, weight_decay=0):
        super(MomentumSGDOptimizer, self).__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.previous_deltas = [0] * len(parameters)
        self.count = 0

    def step(self):
        self.count = 0
        for parameter in self.parameters:
            grad1 = parameter.grad
            grad2 = self.weight_decay*(parameter.data)
            grad3 = self.momentum*self.previous_deltas[self.count]
            grad_tot = grad1+grad2+grad3
            parameter.data = parameter.data - (self.learning_rate*(grad_tot))
            self.previous_deltas[self.count] = grad_tot
            self.count = self.count+1
            
            pass
        
