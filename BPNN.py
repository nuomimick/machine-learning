import numpy as np
from numpy.random import random,permutation

class BPNN:
	"""backpropagation neural networks"""
	def __init__(self,ni,nh,no,lr_o,lr_h,steps,err):
		"""
		lr_o: learning rate of output layer
		lr_h: learing rate of hidden layer
		ni: number of input nodes
		nh: number of hidden nodes
		no: number of output nodes
		"""
		self.ni = ni
		self.nh = nh
		self.no = no
		self.lr_o = lr_o
		self.lr_h = lr_h

		self.__bo = random(self.no) #threshold of output nodes
		self.__who = random((self.no,self.nh))#weight between hidden and output

		self.__bh = random(self.nh) #threshold of hidden nodes
		self.__wih = random((self.nh,self.ni)) #weight between input and hidden

		self.__steps = steps
		self.__err = err


	def fit(self,train_x,train_y):
		for _ in range(self.__steps):
			for idx in permutation(train_x.shape[0]):#SGD
				x = train_x[idx]
				y = train_y[idx] 
				ho, yo = self.__output(x)# output of hidden and output layer
				err = np.sum(np.power(yo - y,2)) / 2
				if err < self.__err:return
				g = yo * (1 - yo) * (y - yo)#gradient of output layer
				e = np.array([ho[i] * (1 - ho[i]) * np.dot(self.__who.T[i],g) for i in range(self.nh)])
				
				for i in range(self.no):
					self.__who[i] += self.lr_o * g[i] * ho
				self.__bo -= self.lr_o * g

				for i in range(self.nh):
					self.__wih[i] += self.lr_h * e[i] * x 
				self.__bh -= self.lr_h * e

	def predict(self,test_x):
		return np.array([list(map(lambda f:1 if f > 0.5 else 0,self.__output(x)[1])) for x in test_x])

	def report(self,predict_y,test_y):
		precision = np.all((predict_y == test_y),axis=1).sum() / len(test_y)
		print(precision)

	def __sigmoid(self,x):
		return 1 / (1 + np.exp(-x))	

	def __output(self,x):
		"""
		x:np.array
		"""	
		o_hidden = self.__sigmoid(np.array([(np.dot(x,self.__wih[i]) - self.__bh[i]) \
					for i in range(self.nh)]))
		#print(o_hidden)
		o_output = self.__sigmoid(np.array([(np.dot(o_hidden,self.__who[i]) - self.__bo[i]) \
					for i in range(self.no)]))
		return o_hidden, o_output

if __name__ == '__main__':
	import numpy as np
	from sklearn.preprocessing import OneHotEncoder
	from sklearn.datasets import load_iris,load_digits
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import Normalizer,MinMaxScaler

	iris = load_iris()
	data = iris.data
	#data = Normalizer().fit_transform(iris.data)
	data = MinMaxScaler((-1,1)).fit_transform(iris.data)
	target = iris.target
	target = OneHotEncoder().fit_transform(iris.target.reshape((-1,1))).toarray()
	train_x,test_x,train_y,test_y = train_test_split(data,target,test_size=0.2)
	bpnn = BPNN(4,10,3,0.1,0.1,50,1e-6)
	bpnn.fit(train_x,train_y)
	bpnn.report(bpnn.predict(test_x),test_y)

	