#!/usr/bin/env python
from math import log
import numpy as np

from bandit.kullback import maxEV
from bandit.Policy import Policy

class KLempUCB(Policy):

    def __init__(self, nbArms, maxReward=1.):
        self.c = 1
        self.nbArms = nbArms
        self.maxReward = maxReward
        self.nbDraws = dict()
        self.obs = dict() 

#    def startGame(self):
        self.t = 1
        for arm in range(self.nbArms):
            self.nbDraws[arm] = 0
            self.obs[arm] = dict({self.maxReward:0})

    def computeIndex(self, arm):
        if self.nbDraws[arm]>0:
            return self.KLucb(self.obs[arm], self.c*log(self.t)/self.nbDraws[arm])
        else:
            return float('+infinity')

    def getReward(self, arm, reward):
        self.nbDraws[arm] += 1
        if self.obs[arm].has_key(reward):
            self.obs[arm][reward] += 1
        else:
            self.obs[arm][reward] = 1
        self.t += 1

    def KLucb(self, obs, klMax):
        p = (np.array(obs.values())+0.)/sum(obs.values())
        v = np.array(obs.keys())
        #print "calling maxEV("+str(p)+", "+str(v)+", "+str(klMax)+")"
        q = maxEV(p, v, klMax)
        #q2 = kbp.maxEV(p, v, klMax)
        #if max(abs(q-q2))>1e-8:
        #    print "ERROR: for p="+str(p)+" ,v = "+str(v)+" and klMax = "+str(klMax)+" : "
        #    print "q = "+str(q)
        #    print "q2 = "+str(q2)
        #    print "_____________________________"
        #print "q = "+str(q)
        return(np.dot(q,v))
