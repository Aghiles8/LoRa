#!/usr/bin/env python
import numpy as np
import os

def G2S(G):
	return G * np.exp(-2 * G)

def dBmTomW(pdBm):
	pmW = 10.0**(pdBm/10.0)
	return pmW

def dBmTonW(pdBm):
	#pfW = 10 ** ((pdBm + 90.0) / 10.0)
	#pmW = 10 ** (pdBm / 10.0)
	pnW = 10.0**((pdBm+90.0)/10.0)
	return pnW

def guards(skewrate):
	gs = [0 for i in PcktLength_SF]
	for sf in range(0,6):
		gs[sf] = math.ceil(1000*skewrate*airtime(sf+7,CodingRate,PcktLength_SF[sf]+LorawanHeader,Bandwidth)*(max(SFdistribution[sf],1.0/Dutycycle)*math.ceil(datasize/(ChanlperSF[sf]*PcktLength_SF[sf]))+(ChanlperSF[sf]-1))) # msec
	return gs

def getDistanceFromPL(pLoss, logDistParams):
	gamma, Lpld0, d0 = logDistParams
	d = d0*(10.0**((pLoss-Lpld0)/(10.0*gamma)))
	return d

def getDistanceFromPL2(pTX, pRX):
	d = d0 * (10 ** ((pTX - pRX - Lpld0) / (10.0 * gamma)))
	return d

def getDistanceFromPower(pTX, pRX, logDistParams):
	return getDistanceFromPL(pTX - pRX, logDistParams)


#import matplotlib.pyplot as plt
#from   seaborn import scatterplot  as scatter
#import mpl_toolkits.mplot3d.axes3d as p3
#from itertools import cycle, islice





