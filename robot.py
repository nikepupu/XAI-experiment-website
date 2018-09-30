import numpy as np
import random
import copy
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# error type (or explanation type)
random.seed(1)
x_types = ["ok", "obs", "dec", "obs_dec"]
init_err_obs = 0.8
init_err_dec = 0.25
init_err = [{"obs":init_err_obs, "ok":1-init_err_obs}, \
	{"dec":init_err_dec, "ok":1-init_err_dec}]
init_prior_obs = 0.5
init_prior_dec = 0.5
init_prior = [{"obs":init_prior_obs, "ok":1-init_prior_obs},\
	{"dec":init_prior_dec, "ok":1-init_prior_dec}]

class resp:
	def __init__(self, y_i):
		self.y_i = y_i
		self.n_resp = len(y_i)

class expl:
	def __init__(self, expl_form):
		self.expl_form = expl_form
		if "_" in self.expl_form:
			self.n_expl = 2
		elif "ok" in self.expl_form:
			self.n_expl = 0
		else:
			self.n_expl = 1
				
class robot:
	def __init__(self, T, err=init_err, prior=init_prior, expl_form=x_types):
		# dict err_type: prob
		self.err = err
		self.n_err = len(self.err)
		self.expl_form = expl_form
		self.hist_resp = []
		self.T = T
		self.steps = 0
		self.exp_num = 0
		self.gamma = 0.5
		# size of memory, 0 means everything
		self.memory_size = 10
		# TO DO
		self.err_type = ["obs", "dec"]
		self.H = {i:0 for i in self.err_type}

		# prior has the same form of err
		self.prior = prior
		self.x_y = []
		for i in range(len(self.prior)):
			x_yi = {}
			for x in self.expl_form:
				x_yi[x] = {}
				for y in self.prior[i]:
					# if no error, picks whatever x
					if y == "ok":
						x_yi[x][y] = 0.25
					# if certain error, picks suitable x
					elif y in x:
						x_yi[x][y] = 0.4
					else:
						x_yi[x][y] = 0.1
			self.x_y.append(x_yi)

		self.yhat_xy = []
		self.acc_with_x = 0.8
		for i in range(self.n_err):
			yhat_xyi = {}
			for yhat in self.prior[i]:
				yhat_xyi[yhat] = {}
				for x in self.expl_form:
					yhat_xyi[yhat][x] = {}
					for y in self.prior[i]:
						# if x is suitable, human prediction would be accurate
						if y in x:
							if yhat == y:
								yhat_xyi[yhat][x][y] = self.acc_with_x
							else:
								yhat_xyi[yhat][x][y] = 1-self.acc_with_x
						# if x is not good, prediction would be less accurate
						else:
							if yhat == y:
								yhat_xyi[yhat][x][y] = 0.5
							else:
								yhat_xyi[yhat][x][y] = 0.5
			self.yhat_xy.append(yhat_xyi)

		# check if there is something wrong
		for i in range(len(self.prior)):
			for y in self.prior[i]:
				cnt = 0
				for x in self.expl_form:
					cnt += self.x_y[i][x][y]
				assert(cnt == 1.0)

		for i in range(self.n_err):
			for x in self.expl_form:
				for y in self.prior[i]:
					cnt = 0
					for yhat in self.prior[i]:
						cnt += self.yhat_xy[i][yhat][x][y]
					assert(cnt == 1.0)

	def sample_action(self):
		# indicate whether certain error occurs
		action = []

		for i in range(self.n_err):
			r = random.random()
			r = 0.9
			prob = self.err[i]["ok"]
			if r > prob:
				action.append(True)
			else:
				action.append(False)

		# print action
		return action

	def generate_exp(self):
		# if first step, pick the most informative explanation
		if self.steps == 0:
			expl_form = self.expl_form[-1]
			init_expl = expl(expl_form)
			self.exp_num += 2
			return init_expl

		expl_form_min = None
		cost_min = 1e5

		# other wise, pick exp with minimum cost
		for e in self.expl_form:

			# # give less exp over time (add some noise to prevent user giving out the same answer)
			# r = random.gauss(0, 0.1)
			# cost = self.cost_trust(e) + ((0.2+r)*(1.8)**(1.0*self.steps/self.T))*self.cost_exp(e)

			# same exp
			cost = self.cost_trust(e) + 0.2*self.cost_exp(e)

			# print "explanation", e, "cost", cost
			if cost < cost_min:
				expl_form_min = e
				cost_min = copy.copy(cost)

		expl_gen = expl(expl_form_min)
		self.exp_num += expl_gen.n_expl

		return expl_gen


	def cost_trust(self, expl_form):
		if len(self.hist_resp) == 0:
			print "Error: history response is empty"
			return 0

		yhat_x = []
		for i in range(self.n_err):
			yhati_x = {}
			for yhat in self.prior[i]:
				yhati_x[yhat] = {}
				for x in self.expl_form:
					yhati_x[yhat][x] = 0
					for y in self.prior[i]:
						yhati_x[yhat][x] += self.yhat_xy[i][yhat][x][y] \
							* self.x_y[i][x][y] * self.prior[i][y]

			yhat_x.append(yhati_x)

		cost_trust = 0

		for t in range(len(self.hist_resp)):
			resp = self.hist_resp[t]
			for i in range(resp.n_resp):
				y_i = resp.y_i[i]
				cost_trust += -np.log(yhat_x[i][y_i][expl_form])

		return 1.0*cost_trust/len(self.hist_resp)

	def cost_exp(self, expl_form):
		if "_" in expl_form:
			return 2.0
		elif "ok" in expl_form:
			return 0.0
		else:
			return 1.0

	def update_mem(self, resp_form):
		self.steps += 1
		response = resp(resp_form)
		self.hist_resp.insert(0, response)
		if len(self.hist_resp) > self.memory_size:
			res_old = self.hist_resp.pop()
			for i in range(res_old.n_resp):
				y_i = res_old.y_i[i]
				if not y_i == "ok":
					self.H[y_i] -= 1

		for i in range(response.n_resp):
			y_i = response.y_i[i]
			if not y_i == "ok":
				self.H[y_i] += 1

	def update_prior(self):
		# self.yhat_xy = []
		# for i in range(self.n_err):
		# 	yhat_xyi = {}
		# 	for yhat in self.prior[i]:
		# 		yhat_xyi[yhat] = {}
		# 		for x in self.expl_form:
		# 			yhat_xyi[yhat][x] = {}
		# 			for y in self.prior[i]:
		# 				# if x is suitable, human prediction would be accurate
		# 				if y in x:
		# 					if yhat == y:
		# 						yhat_xyi[yhat][x][y] = self.acc_with_x
		# 					else:
		# 						yhat_xyi[yhat][x][y] = 1-self.acc_with_x
		# 				# if x is not good, prediction would be less accurate
		# 				# either based on history or random
		# 				else:
		# 					if yhat == "ok":
		# 						yhat_xyi[yhat][x][y] = 1 - self.H[self.err_type[i]]*1.0/len(self.hist_resp)
		# 					else:
		# 						yhat_xyi[yhat][x][y] = self.H[self.err_type[i]]*1.0/len(self.hist_resp)
		# 	self.yhat_xy.append(yhat_xyi)

		# # check if there is something wrong
		# for i in range(len(self.prior)):
		# 	for y in self.prior[i]:
		# 		cnt = 0
		# 		for x in self.expl_form:
		# 			cnt += self.x_y[i][x][y]
		# 		assert(cnt == 1.0)

		for i in range(len(self.err_type)):
			e_type = self.err_type[i]
			self.prior[i][e_type] = (self.gamma*
				self.prior[i][e_type])**(1-1.0*self.H[e_type]/len(self.hist_resp))
			# self.prior[i][e_type] = (self.gamma*
			# 	self.prior[i][e_type])**(1-1.0*self.H[e_type]/self.memory_size)
			self.prior[i]["ok"] = 1-self.prior[i][e_type]

		# print "prior", self.prior
		return self.prior


