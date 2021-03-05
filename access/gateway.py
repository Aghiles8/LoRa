#!/usr/bin/env python
import numpy as np
import copy
from   access.log import tools

class myBS():
	def __init__ (self, params):
		self.params			= params

	def receive(self, ed):
		x   = self.check(ed.packets[self.id])
		ed.H.append(ed.packets[self.id].measure())
		self.clear(ed.packets[self.id])
		if x:
			ed.P.append(ed.packets[self.id].measure())
			self.params.server.receive(ed.packets[self.id])

	def send(self, ed):
		x = self.check(ed.packets[self.id])
		self.clear(ed.packets[self.id])
		return x

	def add(self, pkt):
		for channel, signal in pkt.s.items():
			self.S[channel]             += signal
			self.P[channel][pkt.ed.id]   = pkt
			self.G[pkt.freq][pkt.sf-7]  += pkt.load

	def clear(self, pkt):
		for channel, signal in pkt.s.items():
			self.S[channel] 						-= signal
			self.P[channel].pop(pkt.ed.id)
			self.G[pkt.freq][pkt.sf-7]  -= pkt.load

	def check(self, pkt):
		pkt.lost            = 0
		pkt.collision       = 0
		for tmp in self.P[pkt.freq].values():
			if not tmp.lost:
				a1 					=  tmp.s[pkt.freq][tmp.sf - 7]
				a2 					= self.S[pkt.freq][tmp.sf - 7]

				w1 					= 1 + self.params.capture if self.params.capture !=0 else 1
				w2 					=     self.params.capture if self.params.capture !=0 else 1

				pkt.lost			= 1 if np.any(w1 * a1 < self.params.interaction[tmp.sf - 7] * a2) else 0
				pkt.collision		= 1 if w1 * a1 < w2 * a2 or a1 < a2 else 0

				pkt.x				= not pkt.lost and not pkt.collision
				tmp.lost			= pkt.lost
				tmp.collision		= pkt.collision
		if pkt.x:
			self.tm					= tools.G2S(self.gm)
			self.tm_			    = tools.G2S(self.gm)
#			self.gm				= np.mean([self.G[i][j-7] for i in self.params.freqSet for j in self.params.sfSet])
#			self.tm_				= tools.G2S(self.gm)
			return 1
		else:
			self.tm_				= tools.G2S(self.gm)
			self.gm				= np.mean([self.G[i][j-7] for i in self.params.freqSet for j in self.params.sfSet])
			self.tm				= tools.G2S(self.gm)
		return 0

	def reload(self):
		self.P	     = {freq:{}               for freq in self.params.freqSet}
		self.S	     = {freq:np.zeros((6, 1)) for freq in self.params.freqSet}
		self.G	     = {freq:np.zeros((6, 1)) for freq in self.params.freqSet}
		self.gm      = 0
		self.tm      = 0
		self.tm_     = 0

#		self.T	     = {freq:np.zeros((6, 1)) for freq in self.params.freqSet}
#		self.TH      = list()
#		self.TH.append(self.T)
#		self.TM      = list()
#		self.TM.append(self.tm)
#		self.GH      = list()
#		self.GH.append(self.G)
#		self.GM      = list()
#		self.GM.append(self.gm)
		#		if np.all(np.sqrt(np.sum(np.square([(b.x, b.y) for b in bsDic.values()] - np.array([self.x,self.y]).reshape(1,2)), axis=1)) > params.range*1.5):

