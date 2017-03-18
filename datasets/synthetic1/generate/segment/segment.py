import random

class Segment:

	def __init__(self, length, gauss_mu=0, gauss_sigma=0, step = 1, ignore_base=False, **kwargs):	
		self.kwargs = kwargs	
		self.length = length
		self.step = step		
		self.gauss_mu = gauss_mu
		self.gauss_sigma = gauss_sigma	
		self.ignore_base = ignore_base

	def setGaussMuAndSigma(self, gauss_mu, gauss_sigma):
		self.gauss_mu = gauss_mu
		self.gauss_sigma = gauss_sigma	

	def setStep(self, step):
		self.step = step;

	def generate(self, time, base):
		result = []
		x = 0
		i = 0

		if self.ignore_base:
			base = 0

		while(x <= self.length):
			value = base + self.getValue(x) + random.gauss(self.gauss_mu, self.gauss_sigma)
			result.append([time + i, value])
			i += 1
			x += self.step
		return [result, time + i]