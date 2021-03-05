#!/usr/bin/env python
import numpy	as np
import math
import copy
from   access.log import tools
from   access.log import qos

class myPacket():
	def __init__(self, bs, ed ,params):
		self.params			= params
		self.ed				= ed
		self.bs				= bs
		self.id				= -1
		self.r				= 0
		self.dist			= ed.dist[bs.id]

	def update(self, action):
		self.id				+= 1
		self.lost			= 0
		self.collision		= 0
		self.xxx			= 0
		self.x				= 1
		if self.params.algo != "ADR":
			self.ed.freq, self.ed.sf, self.ed.ptx	, self.ed.bw	, self.ed.cr		= self.ed.setActions[action]
		self.freq   , self.sf   , self.ptx		, self.bw	, self.cr 					= self.ed.freq, self.ed.sf, self.ed.ptx , self.ed.bw	, self.ed.cr

		self.action	= self.ed.action
		self.prx		= qos.getPRX(self.ed.ptx, self.dist, self.params)
		self.s			= qos.getRSSI(self.bs, self.prx, self.ed.sf, self.ed.freq, self.ed.bw)
		self.toa		= qos.getToA(self.ed)
		self.etx		= qos.getEnergy(self.toa, self.ed.ptx)
		self.snr		= qos.getSNR(self.ed.bw, self.prx)
		self.ber		= qos.getBER(self.ed, self.snr)
#		self.ber		= qos.getBER_(self.ed)
		self.dr		= qos.getDR(self.ed)
		self.load		= qos.getG(self.params.period_mn, self.toa)
		return copy.copy(self)

	def measure(self):
		self.G		 = self.bs.gm
		self.T		 = self.bs.tm
#		tmp = (self.ber * self.etx * self.toa)
#		tmp = 0.0000003 if tmp == 0 else tmp
		self.r		 = self.bs.tm - self.bs.tm_ #* self.dr #/ tmp

		self.prx_mean	 = np.mean([pkt.prx		for pkt in self.ed.H]) if self.id !=0 and self.id !=1 else self.prx
		self.toa_mean	 = np.mean([pkt.toa		for pkt in self.ed.H]) if self.id !=0 and self.id !=1 else self.toa
		self.ptx_mean    = np.mean([pkt.etx		for pkt in self.ed.H]) if self.id !=0 and self.id !=1 else self.etx
		self.ber_mean	 = np.mean([pkt.ber		for pkt in self.ed.H]) if self.id !=0 and self.id !=1 else self.ber
		self.snr_mean	 = np.mean([pkt.snr		for pkt in self.ed.H]) if self.id !=0 and self.id !=1 else self.snr
		self.dr_mean	 = np.mean([pkt.dr		for pkt in self.ed.H]) if self.id !=0 and self.id !=1 else self.dr
		self.pdr_mean	 = np.mean([pkt.x 		for pkt in self.ed.H]) if self.id !=0 and self.id !=1 else self.x
		self.T_mean 	 = np.mean([pkt.T 		for pkt in self.ed.H]) if self.id !=0 and self.id !=1 else self.T
		self.G_mean 	 = np.mean([pkt.G 		for pkt in self.ed.H]) if self.id !=0 and self.id !=1 else self.G
		self.r_mean    = np.mean([pkt.r			for pkt in self.ed.H]) if self.id !=0 and self.id !=1 else self.r

#		print(self.params.env.now/216000000)
		return copy.copy(self)







