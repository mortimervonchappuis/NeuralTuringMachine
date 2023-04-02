import tensorflow as tf



class Head(tf.keras.layers.Layer):
	def __init__(self, 
				 n_memory_units, 
				 n_shifts, 
				 **kwargs):
		super().__init__(**kwargs)
		self.n_memory_units = n_memory_units
		self.n_shifts       = n_shifts * 2 + 1
		self.key	        = tf.keras.layers.Dense(n_memory_units,   kernel_initializer='LecunNormal', activation='tanh',     name='key')
		self.shift          = tf.keras.layers.Dense(n_shifts * 2 + 1, kernel_initializer='LecunNormal', activation='softmax',  name='shift')
		self.beta           = tf.keras.layers.Dense(1,                kernel_initializer='LecunNormal', activation='softplus', name='beta')
		self.gate           = tf.keras.layers.Dense(1,                kernel_initializer='LecunNormal', activation='sigmoid',  name='gate')
		self.gamma          = tf.keras.layers.Dense(1,                kernel_initializer='LecunNormal', activation='softplus', name='gamma')


	@tf.function
	def address(self, 
				Z, 
				W_old, 
				M):
		key   = self.key(Z)
		shift = self.shift(Z)
		beta  = self.beta(Z)
		gate  = self.gate(Z)
		gamma = self.gamma(Z) + 1
		# COMPUTE
		W_new = self.content(beta, key, M)
		W_new = self.interpolate(gate, W_old, W_new)
		W_new = self.location(shift, W_new)
		W_new = self.sharpen(gamma, W_new)
		return W_new


	@tf.function
	def K(self, 
		  key, 
		  M):
		# K = (B, N); M = (B, M, N)
		K_norm = tf.sqrt(tf.reduce_sum(tf.square(key), axis=1))[...,None]
		M_norm = tf.sqrt(tf.reduce_sum(tf.square(M), axis=2))
		return tf.matmul(M, key[...,None])[...,0] / (K_norm * M_norm + 1e-8)


	@tf.function
	def content(self, 
				beta, 
				key, 
				M):
		# key = (B, N); M = (M, N); Z = (B, M)
		Z = beta * self.K(key, M)
		return tf.nn.softmax(Z, axis=1)


	@tf.function
	def interpolate(self, 
					gate, 
					W_old, 
					W_new):
		# gate = (B); W_old = (B, M); W_new = (B, M)
		gate = tf.expand_dims(gate, axis=1)
		W = gate * W_old + (1 - gate) * W_new
		return W[:,0,:]


	@tf.function
	def location(self, 
				 shift, 
				 W):
		# shift = (B, S); W = (B, M)
		S = tf.stack([tf.roll(W, n - self.n_shifts//2, axis=1) for n in range(self.n_shifts)], axis=1)
		return tf.reduce_sum(S * tf.expand_dims(shift, axis=2), axis=1)


	@tf.function
	def sharpen(self, 
				gamma,
				W):
		# gamma = (B); W = (B, M)
		W = tf.math.pow(W, gamma)
		return W/tf.reduce_sum(W, axis=1)[...,None]



class ReadHead(Head):
	def __init__(self, n_memory_units, n_shifts, **kwargs):
		super().__init__(n_memory_units, n_shifts, **kwargs)


	@tf.function
	def call(self, Z, W, M):
		W = self.address(Z, W, M)
		return tf.matmul(tf.expand_dims(W, axis=1), M)[:,0,:], W



class WriteHead(Head):
	def __init__(self, n_memory_units, n_shifts, **kwargs):
		super().__init__(n_memory_units, n_shifts, **kwargs)
		self.add   = tf.keras.layers.Dense(n_memory_units, kernel_initializer='LecunNormal', activation='tanh')
		self.erase = tf.keras.layers.Dense(n_memory_units, kernel_initializer='LecunNormal', activation='sigmoid')


	@tf.function
	def call(self, Z, W, M):
		W = tf.expand_dims(self.address(Z, W, M), axis=2)
		A = tf.expand_dims(self.add(Z),           axis=1)
		E = tf.expand_dims(self.erase(Z),         axis=1)
		return W, A, E



if __name__ == '__main__':
	batch_size, n_memory_units, n_memory_slots, n_shifts = 2, 8, 4, 1
	W_old = tf.constant( [[0.0, 0.0, 1.0, 0.0],
						  [0.0, 0.5, 0.5, 0.0]])
	Z     = tf.constant( [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
						  [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]])
	M     = tf.constant([[[0.0, 1.5, 1.5, 1.5, 0.0, 1.5, 1.5, 1.5],
						  [1.2, 0.0, 1.2, 1.2, 0.0, 1.2, 1.2, 1.2],
						  [1.1, 1.1, 0.0, 1.1, 0.0, 1.1, 1.1, 1.1],
						  [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
						 [[0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5],
						  [0.2, 0.2, 0.2, 0.0, 0.2, 0.0, 0.2, 0.2],
						  [0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1],
						  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
	key   = tf.constant( [[0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.0, 0.0],
						  [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.0, 0.0]])
	shift = tf.constant( [[1.0, 0.0, 0.0],
						  [0.0, 0.0, 1.0]])
	beta  = tf.constant( [[1.0],
						  [3.0]])
	gate  = tf.constant( [[0.0, 0.3, 0.7, 1.0],
						  [0.4, 0.3, 0.2, 0.1]])
	gamma = tf.constant( [[2.5],
						  [1.0]])
	erase = tf.constant( [[0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
						  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
	add   = tf.constant( [[0.0, 0.0, 0.0, 0.0, 1.0,-1.0, 0.0, 0.0],
						  [0.0, 0.0,-0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])
	head = Head(n_memory_units, n_shifts)
	W_new = head.content(beta, key, M)
	#print('content', W_new)
	W = head.interpolate(gate, W_old, W_new)
	print('shift', shift)
	print('interpolate', W)
	W = head.location(shift, W)
	print('location', W)
	W = head.sharpen(gamma, W)
	#print('sharpen', W)
	read = ReadHead(n_memory_units, n_shifts)
	R, W = read(Z, W_old, M)
	#print('W', W)
	#print('R', R)
	#print('M', M)
	#print('erase', erase)
	#print('add', add)
	M = M * (1 - tf.matmul(tf.expand_dims(W, axis=2), tf.expand_dims(erase, axis=1)))
	#print('M erase', M)
	M = M + tf.matmul(tf.expand_dims(W, axis=2), tf.expand_dims(add, axis=1))
	#print('M add', M)
