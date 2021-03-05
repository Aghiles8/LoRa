#!/usr/bin/env python
from bandit.kullback import klucbGauss
from bandit.klUCB import klUCB

class UCB(klUCB):
    def __init__(self, nbArms, amplitude=1., lower=0.):
        klUCB.__init__(self, nbArms, amplitude, lower, lambda x, d, sig2: klucbGauss(x, d, .25))


