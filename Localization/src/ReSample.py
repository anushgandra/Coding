#!/usr/bin/env python

import rospy
import numpy as np
from threading import Lock

'''
  Provides methods for re-sampling from a distribution represented by weighted samples
'''
class ReSampler:

  '''
    Initializes the resampler
    particles: The particles to sample from
    weights: The weights of each particle
    state_lock: Controls access to particles and weights
  '''
  def __init__(self, particles, weights, state_lock=None):
    self.particles = particles 
    self.weights = weights
    
    # Indices for each of the M particles
    self.particle_indices = np.arange(self.particles.shape[0])  
    
    # Bins for partitioning weight values
    self.step_array = (1.0/self.particles.shape[0]) * np.arange(self.particles.shape[0], dtype=np.float32)
    
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock
  
  '''
    Student's IMPLEMENT
    Performs in-place, lower variance sampling of particles
    returns: nothing
  '''
  def resample_low_variance(self):
    self.state_lock.acquire()
    
    # YOUR CODE HERE
    M = len(self.particle_indices)
    new_particles = []
    r = np.random.uniform(0,1.0/(M))
    c = self.weights[0]
    i = 0
    for m in self.particle_indices:
        U = r+(m*(1.0/M))
	while(U > c):
            i = i + 1
	    c = c + self.weights[i]
	new_particles.append(self.particles[i,:])
    self.particles[:] = np.array(new_particles)[:]
    self.weights[:] = 1.0 / self.particles.shape[0]
    self.state_lock.release()
    
    
    

