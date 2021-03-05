#!/usr/bin/env python
from random import choice
from bandit.Policy import Policy

class Thompson(Policy):

  def __init__(self, nbArms, posterior):
    self.nbArms = nbArms
    self.posterior = dict()
    for arm in range(self.nbArms):
      self.posterior[arm] = posterior()

#  def startGame(self):
    self.t = 1;
    for arm in range(self.nbArms):
      self.posterior[arm].reset()

  def getReward(self, arm, reward):
    self.posterior[arm].update(reward)
    self.t += 1
    
  def computeIndex(self, arm):
    return self.posterior[arm].sample()
