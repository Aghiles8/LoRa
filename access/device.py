#!/usr/bin/env python
import numpy 				as np
from   access.log           import tools
from   access.packet 		import myPacket

class myED():
	def __init__(self, params):
		self.params				= params
		self.H						= list()
		self.P						= list()
		self.setActions 	= [(params.freqSet[i], params.sfSet[j], params.powerSet[k], params.bwSet[l], params.crSet[m]) for i in range(len(params.freqSet)) for j in range(len(params.sfSet)) for k in range(len(params.powerSet)) for l in range(len(params.bwSet)) for m in range(len(params.crSet))]
		self.actions			= len(self.setActions)

	def send(self, bsDict):
		for bsid, bs in bsDict.items():
			bs.add(self.packets[bsid].update(self.action))
		self.time = self.packets[bsid].toa
		return self.time

	def receive(self):
		if self.params.algo == "random" or self.params.algo == "ADR":
			self.action			= np.random.choice(self.actions, p=self.policy[self.edapp])
		elif self.bestbs.send(self):
			self.action			= self.newaction
			self.app			= self.newapp

	def reload(self):
		self.H						= list()
		self.P						= list()
		self.reward				= np.zeros((self.actions, 3))
		self.weights			= np.ones (self.actions)

#		self.policy				= np.random.rand(self.actions)
#		self.policy				= [self.policy[x]/sum(self.policy) for x in range(0, self.actions)]
#		self.q_table			= np.zeros((3, self.actions))
		self.policy				= np.zeros((3, self.actions))
		for i in range(3):
			self.policy[i]	= [1/self.actions for x in range(self.actions)]    #np.random.rand(self.actions)
			#self.policy[i]	= [self.policy[i][x]/sum(self.policy[i]) for x in range(self.actions)] # [ for x in range(self.actions)]   

		self.app					= self.edapp
		self.newapp					= self.edapp
		self.smart					= 1
		self.sf  					= 12
		self.freq					= 868100
		self.ptx 					= 14
		self.bw  					= 125
		self.cr 					= 2
		self.fs						= 0.1
		self.ps2 					= 8
		self.ps3 					= 2
		self.hdr 					= True
		self.crc 					= True
		self.dist    				= {e.id:np.sqrt((e.x - self.x)**2 + (e.y - self.y)**2) for e in self.params.bsDict.values()}
		self.packets 				= {bsid:myPacket(bs, self, self.params) for bsid, bs in self.params.bsDict.items()}
#		mm                = self.getMemberships() if self.id==0 else mm
#		self.alpha      	= copy.deepcopy(mm)
#		self.reward     	= copy.deepcopy(mm)
#		self.algo					= self.params.algo
		self.period			= self.params.period
		self.ps1 				= self.params.ps
		self.action			= np.random.choice(self.actions, p=self.policy[self.app])
		self.newaction			= self.action
#		tools.createFile(self.params.path+"/"+str(self.edapp)+"_"+str(self.id)+".csv"       , logs.gettext1(self))



#self.reward				= np.zeros((self.actions, 3))
#self.weights			= np.ones (self.actions)
#self.policy				= np.zeros((3, self.actions))
#for i in range(3):
#	#self.policy[i]	= np.random.rand(self.actions)
#	self.policy[i]	= [1/3 for x in range(self.actions)]  #       [self.policy[i][x]/sum(self.policy[i]) for x in range(self.actions)]
