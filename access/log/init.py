#!/usr/bin/env python
import numpy as np
import math
import random
import os
import copy
import pandas           as pd
from   os.path 		import join, exists
from   os 				import makedirs
from   access.gateway	import myBS
from   access.device	import myED
from   random 			import randint
from   access.log		import qos

def createBS(params, grid):
	xRange   = [grid[0]*0.1, grid[0]*0.9]
	yRange   = [grid[1]*0.1, grid[1]*0.9]
	bsDic    = {}
	for n in range(params.nrBS):
		bs         = myBS(params)
		bs.id      = n
		bs.x       = random.uniform(xRange[0], xRange[1]) if params.nrBS != 1 else grid[0]*0.5
		bs.y       = random.uniform(yRange[0], yRange[1]) if params.nrBS != 1 else grid[1]*0.5
		bsDic[bs.id] = bs
	return bsDic

def createED(params, bsDict, grid):
	distMatrix  = qos.getMaxTransmitDistance(params, (1, 90, 8, 4.25, False, True))
	edDict      = {}
	xRange    	= [0, grid[0]]
	yRange    	= [0, grid[1]]
	temp 	  	  = 0
	for idx in range(len(params.distribution)):
		number_nodes = int(params.nrED * params.distribution[idx])
		for n in range(number_nodes):
			while True:
				x   = random.uniform(xRange[0], xRange[1])
				y   = random.uniform(yRange[0], yRange[1])
				tmp = np.sum(np.square([(bs.x, bs.y) for bs in bsDict.values()] - np.array([x,y]).reshape(1,2)), axis=1)
				if np.any(tmp <= params.range**2) and np.any(tmp <= distMatrix[idx]**2):
					ed				= myED(params)
					ed.id  			= n+temp
					ed.drx			= (5-idx)
					ed.x   			= x
					ed.y   			= y
					ed.edapp		= randint(0, 2)
					edDict[ed.id] 	= ed
					break
		temp += number_nodes
	#plot_Locations(bsDict, edDict, grid[0], grid[1], distMatrix)
	return edDict



def createNetwork(params):
	params.grid			= params.range * params.nrBS
	grid				= [params.grid, params.grid]
	params.bsDict		= createBS(params, grid)
	params.edDict		= createED(params, params.bsDict, grid)




