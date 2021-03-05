#!/usr/bin/env python
from math import sqrt, log
from bandit.Policy import Policy

class UCBV(Policy):
    def __init__(self, nbArms, amplitude=1., lower=0.):
        self.nbArms = nbArms
        self.amplitude = amplitude
        self.lower = lower
        self.nbDraws = dict()
        self.cumReward = dict()
        self.cumReward2 = dict()

#    def startGame(self):
        self.t = 1
        for arm in range(self.nbArms):
            self.nbDraws[arm] = 0
            self.cumReward[arm] = 0.0
            self.cumReward2[arm] = 0.0

    def computeIndex(self, arm):
        if self.nbDraws[arm] < 2:
            return float('+infinity')
        else:
            m = self.cumReward[arm]/self.nbDraws[arm] 
            v = self.cumReward2[arm]/self.nbDraws[arm] - m*m 
            return m + sqrt(2*log(self.t) * v / self.nbDraws[arm]) + 3*self.amplitude*log(self.t)/self.nbDraws[arm]

    def getReward(self, arm, reward):
        self.nbDraws[arm] += 1
        self.cumReward[arm] += reward
        self.cumReward2[arm] += reward**2
        self.t += 1
