#!/usr/bin/python3

import numpy as np
import random
import argparse
import sys
import fileinput
import csv
import simpy
from   access.log 		import qos, init
from   access.server	import myServer


class params():
	pass

params.range = 14000
params.d0 = 40.0
params.gamma = 2.2394
params.Lpld0 = 95.0038
params.GL = 0
params.distribution  = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]

params.algo = "q-learning" #, "markov", "exp3", "random", "bayesUCB", "thompson", "UCB", "klUCB"])
params.nrBS = 1
params.nrED = 100
params.ps  = 10
params.period_mn = 1

params.bwSet  = [125,250,500]
params.powerSet = [11,12,13,14]
params.crSet  = [1,2,3,4]
params.sfSet  = [7, 8, 9, 10, 11, 12]
params.freqSet = [868100]

params.capture, params.interaction              = qos.getInteractionMatrix(True, True)
params.nrPkt  = 50 #650 # 2min #3.58j chaque 4min
params.snrx   = [-20,-17.5,-15,-12.5,-10,-7.5]
params.drx    = [0.25,0.44,0.98,1.76,3.125,5.47]
params.sensi = qos.getSensi()

class a():
	pass

results		= a()
results.prx	= list()
results.toa	= list()
results.ptx	= list()
results.ber	= list()
results.snr	= list()
results.dr 	= list()
results.T		= list()
results.G   = list()
results.pdr = list()
results.r   = list()
results.all = np.zeros((3,10,10,10,9))

def sim_transmit(env, ed, bsDict, server):
	while True:
		yield env.timeout(random.expovariate(1/ed.period))
		yield env.timeout(ed.send(bsDict))
		for bsid, bs in bsDict.items():
			bs.receive(ed)
		yield env.timeout(server.send(ed))
		ed.receive()
		yield env.timeout(ed.period - ed.time - server.time)

def run(params):
	params.period	  = int(params.period_mn*60*1000)
	params.sim_time = int(params.period*params.nrPkt*2)

	for bs in params.bsDict.values():
		bs.reload()
	for ed in params.edDict.values():
		ed.reload()
	params.server		= myServer(params)
	params.env      = simpy.Environment()
	for ed in params.edDict.values():
		params.env.process(sim_transmit(params.env, ed, params.bsDict, params.server))
	params.env.run(until=params.sim_time)

	mini = min([len(params.edDict[i].H) for i in range(params.nrED)])
	for j in range(3):
		results.pdr.append([np.mean([ed.H[i].pdr_mean for ed in params.edDict.values() if ed.edapp == j]) for i in range(mini)])
		results.toa.append([np.mean([ed.H[i].toa_mean for ed in params.edDict.values() if ed.edapp == j]) for i in range(mini)])
		results.ber.append([np.mean([ed.H[i].ber_mean for ed in params.edDict.values() if ed.edapp == j]) for i in range(mini)])
		results.ptx.append([np.mean([ed.H[i].ptx_mean for ed in params.edDict.values() if ed.edapp == j]) for i in range(mini)])
		results.prx.append([np.mean([ed.H[i].prx_mean for ed in params.edDict.values() if ed.edapp == j]) for i in range(mini)])
		results.snr.append([np.mean([ed.H[i].snr_mean for ed in params.edDict.values() if ed.edapp == j]) for i in range(mini)])
		results.dr.append ([np.mean([ed.H[i].dr_mean  for ed in params.edDict.values() if ed.edapp == j]) for i in range(mini)])
		results.T.append  ([np.mean([ed.H[i].T_mean   for ed in params.edDict.values() if ed.edapp == j]) for i in range(mini)])
		results.G.append  ([np.mean([ed.H[i].G_mean   for ed in params.edDict.values() if ed.edapp == j]) for i in range(mini)])
		results.r.append  ([np.mean([ed.H[i].r_mean   for ed in params.edDict.values() if ed.edapp == j]) for i in range(mini)])

	results.pdr.append([np.mean([ed.H[i].pdr_mean for ed in params.edDict.values()]) for i in range(mini)])
	results.toa.append([np.mean([ed.H[i].toa_mean for ed in params.edDict.values()]) for i in range(mini)])
	results.ber.append([np.mean([ed.H[i].ber_mean for ed in params.edDict.values()]) for i in range(mini)])
	results.ptx.append([np.mean([ed.H[i].ptx_mean for ed in params.edDict.values()]) for i in range(mini)])
	results.prx.append([np.mean([ed.H[i].prx_mean for ed in params.edDict.values()]) for i in range(mini)])
	results.snr.append([np.mean([ed.H[i].snr_mean for ed in params.edDict.values()]) for i in range(mini)])
	results.dr.append ([np.mean([ed.H[i].dr_mean  for ed in params.edDict.values()]) for i in range(mini)])
	results.T.append  ([np.mean([ed.H[i].T_mean   for ed in params.edDict.values()]) for i in range(mini)])
	results.G.append  ([np.mean([ed.H[i].G_mean   for ed in params.edDict.values()]) for i in range(mini)])
	results.r.append  ([np.mean([ed.H[i].r_mean   for ed in params.edDict.values()]) for i in range(mini)])

	print(results.pdr)

init.createNetwork(params)
run(params)

