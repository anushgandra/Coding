import numpy as np

class Sampler:
    def __init__(self, env):
        self.env = env
        self.xlimit = self.env.xlimit
        self.ylimit = self.env.ylimit

    def array_in(self, arr, list_of_arr):
     	for elem in list_of_arr:
        	if (arr == elem).all():
            		return True
     	return False

    def sample(self, num_samples):
        """
        Samples configurations.
        Each configuration is (x, y).

        @param num_samples: Number of sample configurations to return
        @return 2D numpy array of size [num_samples x 2]
        """
	samples = []
	while(np.shape(samples)[0]!=num_samples):
		x = np.random.uniform(self.xlimit[0],self.xlimit[1])
		x = x - x%self.env.stepsize
		y = np.random.uniform(self.ylimit[0],self.ylimit[1])
		y = y - y%self.env.stepsize
		cur = np.array([x,y])
		col = self.env.state_validity_checker(cur)[0]
		if(col==1 and not self.array_in(cur,samples)):
			samples.append(cur)
        return np.array(samples)

    
