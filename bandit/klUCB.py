#!/usr/bin/env python
from math               import log
from bandit.Policy import Policy
from bandit import kullback

class klUCB(Policy):

    def __init__(self, nbArms, amplitude=1., lower=0., klucb=kullback.klucbBern):
        self.c         = 1.
        self.nbArms    = nbArms
        self.amplitude = amplitude
        self.lower     = lower
        self.nbDraws   = dict()
        self.cumReward = dict()
        self.klucb     = klucb

#    def startGame(self):
        self.t = 1
        for arm in range(self.nbArms):
            self.nbDraws[arm] = 0
            self.cumReward[arm] = 0.0

    def computeIndex(self, arm):
        if self.nbDraws[arm] == 0:
            return float('+infinity')
        else:
            return self.klucb(self.cumReward[arm] / self.nbDraws[arm], self.c * log(self.t) / self.nbDraws[arm], 1e-4) # Could adapt tolerance to the value of self.t

    def getReward(self, arm, reward):
        self.nbDraws[arm] += 1
        self.cumReward[arm] += (reward - self.lower) / self.amplitude
        self.t += 1


    # Debugging code
    #print "arm " + str(arm) + " receives " + str(reward)
    #print str(self.nbDraws[arm]) + " " + str(self.cumReward[arm])






