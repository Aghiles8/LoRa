#!/usr/bin/env python
import numpy as np
import random
import math
import csv
from numpy             import loadtxt
from numpy             import savetxt
from access.log        import tools
from fcmeans           import FCM
from bandit.Beta       import Beta
from bandit.UCB        import UCB
from bandit.UCBV       import UCBV
from bandit.klUCB      import klUCB
from bandit.KLempUCB   import KLempUCB
from bandit.Thompson   import Thompson
from bandit.BayesUCB   import BayesUCB

class myServer:
	def __init__(self, params):
		self.params						= params
		self.grid						= np.zeros((self.params.nrED, self.params.nrBS))
		self.bayesUCB					= BayesUCB(len(self.params.edDict[0].setActions),Beta)
		self.thompson					= Thompson(len(self.params.edDict[0].setActions),Beta)
		self.UCBV						= UCBV    (len(self.params.edDict[0].setActions),Beta)
		self.UCB						= UCB     (len(self.params.edDict[0].setActions))
		self.klUCB						= klUCB   (len(self.params.edDict[0].setActions))
		self.memberships, self.labels = self.getMemberships()

	def receive(self, pkt):
		self.grid[pkt.ed.id][pkt.bs.id] = pkt.prx * -1

	def send(self, ed):
		ed.bestbs						= self.params.bsDict[np.argmin(self.grid[ed.id])]
		self.grid[ed.id] 				= np.zeros(self.params.nrBS)
		ed.reward[ed.action][ed.app]	= ed.packets[ed.bestbs.id].r

		if self.params.algo   == "Random":
			self.update_random(ed)
		if self.params.algo   == "Markov":
			self.update_markov(ed)
		elif self.params.algo == "EXP3":
			self.update_bandit_exp3(ed)
		elif self.params.algo == "Q-learning":
			self.update_qlearning(ed)
		elif self.params.algo == "ADR":
			self.update_adr(ed)
		elif self.params.algo == "bayesUCB":
			self.update_bandit(ed, self.bayesUCB)
		elif self.params.algo == "Thompson":
			self.update_bandit(ed, self.thompson)
		elif self.params.algo == "UCB":
			self.update_bandit(ed, self.UCB)
		elif self.params.algo == "UCBV":
			self.update_bandit(ed, self.UCBV)
		elif self.params.algo == "klUCB":
			self.update_bandit(ed, self.klUCB)

		ed.bestbs.add(ed.packets[0])
		self.time = ed.packets[ed.bestbs.id].toa
		return self.time

	def update_bandit(self, ed, policy):
		policy.getReward(ed.action, ed.reward[ed.action][ed.app])

		ed.policy[ed.app]					= [policy.computeIndex(i) for i in range(ed.actions)]                            #np.random.choice([arm for arm in ed.policy[ed.app].keys() if ed.policy[ed.app][arm] == max (ed.policy[ed.app].values())])
		ed.policy[ed.app]			    	= [0 if math.isinf(ed.policy[ed.app][i]) else ed.policy[ed.app][i] for i in range(ed.actions)]  #np.random.choice([arm for arm in ed.policy[ed.app].keys() if ed.policy[ed.app][arm] == max (ed.policy[ed.app].values())])

		ed.policy[ed.app]					= [ed.policy[ed.app][x]/sum(ed.policy[ed.app]) for x in range(0, ed.actions)]
		ed.newaction					    = np.random.choice(ed.actions, p=ed.policy[ed.app])
		ed.newapp							= np.argmax(self.memberships[ed.newaction])

	def update_markov(self, ed):
		discount                            = .99
		tau                                 = .1
		epsilon                             = .0001
		values                              = np.zeros(3)
		while True:
			oldValues                       = np.copy(values)
			values[ed.app]                  = ed.reward[ed.action][ed.app] + np.max(discount * np.dot(self.memberships[:][:], values))
			if np.max(np.abs(values - oldValues)) <= epsilon:
				break
		policies                            = np.zeros([3, ed.actions])
		policies[ed.app]                    = np.zeros(ed.actions)
		policies[ed.app]                    = [ed.reward[j][ed.app] for j in range(ed.actions)] + discount * np.dot(self.memberships[:][:], values)
		policies[ed.app]                   -= np.max(policies[ed.app])
		policies[ed.app]                    = np.exp(policies[ed.app] / float(tau))
		policies[ed.app]                   /= policies[ed.app].sum()

		ed.policy[ed.app]                   = policies[ed.app]
		#policy                              = np.random.choice(policies[ed.app], p=ed.policy[ed.app]) #np.random.choice(np.array(np.where(policies[ed.app][:] == policy)).ravel())
		ed.newaction						= np.random.choice(ed.actions, p=ed.policy[ed.app])
		ed.newapp							= np.argmax(self.memberships[ed.newaction])

	def update_bandit_exp3(self, ed):
		alpha								= 0.5
		ed.weights							= [(ed.weights[j] * np.exp((alpha * ed.reward[j][ed.app])/ed.actions)) for j in range(0, ed.actions)]
		ed.policy[ed.app]					= [(1-alpha) * (ed.weights[j]/sum(ed.weights)) + (alpha/ed.actions) for j in range(0, ed.actions)]
		ed.newaction						= np.random.choice(ed.actions, p=ed.policy[ed.app])
		ed.newapp							= np.argmax(self.memberships[ed.newaction])

	def update_qlearning(self, ed):
		gamma								= 0.9
		alpha								= 0.9
#		alpha								= self.memberships[ed.action][ed.app]
		ed.policy[ed.app][ed.action]		= (1 - alpha) * ed.policy[ed.app, ed.action] + alpha * (ed.reward[ed.action][ed.app] + gamma * np.max(ed.policy[ed.app]))
		ed.newaction						= random.randint(0, ed.actions-1) if np.random.uniform(0, 1) < alpha else np.argmax(ed.policy[ed.app])
		ed.newapp							= np.argmax(self.memberships[ed.newaction])

	def update_random(self, ed):
#		prob								= (1/ed.actions) * np.ones(ed.actions)
#		prob								= np.array(prob)
#		prob[prob<0.0005]					= 0
#		ed.policy[ed.app]					= prob/sum(prob)
		ed.newaction						= np.random.choice(ed.actions, p=ed.policy[ed.app])
		ed.newapp							= np.argmax(self.memberships[ed.newaction])

	def getMemberships(self):
		self.params.edDict[0].packets[0].dist = 7000
		param				= {action:self.params.edDict[0].packets[0].update(action) for action in range(len(self.params.edDict[0].setActions))}
		self.params.edDict[0].packets[0].id = -1
		metrics 			= np.array([(pkt.toa/1000, pkt.ber, pkt.snr, pkt.prx, pkt.dr) for pkt in param.values()], dtype=np.float32)

		fcm 				= FCM(n_clusters=3,max_iter=150,m=2,error=1e-5,random_state=42).fit(metrics)
		labels				= [np.argmax(fcm.u[x]) for x in range(0, len(self.params.edDict[0].setActions))]

#		i,\
#		self.params.edDict[0].setActions[i][0],\
#		self.params.edDict[0].setActions[i][1],\
#		self.params.edDict[0].setActions[i][2],\
#		self.params.edDict[0].setActions[i][3],\
#		self.params.edDict[0].setActions[i][4],\
#		tmp[i][0],\
#		tmp[i][1],\
#		tmp[i][2],\
#		tmp[i][3],\
#		tmp[i][4],\
#		fcm.u[i][0],\
#		fcm.u[i][1],\
#		fcm.u[i][2],\
#		labels[i],\

#		savetxt(self.params.topopath+'0_fcm_u.csv', fcm.u , delimiter=',',  fmt='%1.3f')
#		savetxt(self.params.topopath+'0_fcm_l.csv', labels, delimiter=',',  fmt='%01d')
#		savetxt(self.params.topopath+'0_fcm_a.csv', self.params.edDict[0].setActions, delimiter=',', fmt='%04.1f,%01d,%02d,%03d,%01d')
#		savetxt(self.params.topopath+'0_fcm_m.csv', metrics, delimiter=',', fmt='%05.4f')
		return fcm.u, labels

	def update_adr(self, ed):
		snr = -300
		for pkt in ed.P[-20:]:
			snr = max(snr, pkt.snr)
		ii  = snr - self.params.snrx[ed.drx] - 5
		ii /= 3
		ii  = int(ii)
		while ii != 0:
			if ii < 0:
				if ed.ptx < 12:
					ed.ptx +=3
				else:
					break
				ii+=1
			else:
				if ed.packets[ed.bestbs.id].dr > self.params.drx[ed.drx]:
					if ed.ptx > 13:
						ed.ptx -=3
					else:
						break
				elif ed.sf > 7:
					ed.sf-=1
				ii-=1

#		self.packets[0].id = 0
#		self.bsDict[0].H   = list()
#		self.bsDict[0].H.append({freq:np.zeros((6, 1)) for freq in self.params.freqSet})
#		self.bsDict[0].M   = list()
#		self.bsDict[0].M.append(0)


