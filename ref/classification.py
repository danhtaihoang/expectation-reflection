class model(object):

	def __init__(self,max_iter=100,regu=0.001):
		self.max_iter = max_iter
		self.regu = regu

	def fit(self,x,y):
		w = y/x
		b = y - w*x
		
		self.b = b
		self.w = w
		
		# just for output in the main script
		self.intercept_ = b
		self.coef_ = w

		return self

	def predict(self,x):
		b = self.b
		w = self.w

		y = b + w*x

		return y


