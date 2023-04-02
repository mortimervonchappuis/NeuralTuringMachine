import tensorflow as tf
import numpy as np
import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt
from heads import *




class NeuralTuringMachineCell(tf.keras.layers.AbstractRNNCell):
	def __init__(self, 
				 ctrl_class=tf.keras.layers.LSTMCell, 
				 n_ctrl_layers=1, 
				 n_ctrl_units=100, 
				 n_output_units=8, 
				 n_memory_slots=128, 
				 n_memory_units=20, 
				 n_read_heads=1, 
				 n_write_heads=1, 
				 n_shifts=1, 
				 clip_value=20, 
				 **kwargs):
		super().__init__(**kwargs)
		self.clip_value     = clip_value
		self.n_memory_units = n_memory_units
		self.n_memory_slots = n_memory_slots
		self.n_output_units = n_output_units
		self.n_read_heads   = n_read_heads
		self.n_write_heads  = n_write_heads
		self.ctrl           = tf.keras.layers.StackedRNNCells([ctrl_class(n_ctrl_units) for _ in range(n_ctrl_layers)])
		self.read_heads     = [ReadHead(n_memory_units, n_shifts)  for _ in range(n_read_heads)]
		self.write_heads    = [WriteHead(n_memory_units, n_shifts) for _ in range(n_write_heads)]
		self.out            = tf.keras.layers.Dense(n_output_units, kernel_initializer='LecunNormal')
		# INIT
		Memory_init            = tf.keras.initializers.Constant(1e-6)(shape=(1, self.n_memory_slots, self.n_memory_units), dtype=tf.float32)
		read_init              = tf.random.normal(shape=(1, self.n_memory_units * self.n_read_heads,), dtype=tf.float32)
		weight_read_init       = tf.random.normal(shape=(1, self.n_read_heads,  self.n_memory_slots),  dtype=tf.float32)
		weight_write_init      = tf.random.normal(shape=(1, self.n_write_heads, self.n_memory_slots),  dtype=tf.float32)
		self.Memory_init       = tf.constant(Memory_init,       name='Memory_init')
		self.read_init         = tf.Variable(read_init,         name='read_init')
		self.weight_read_init  = tf.Variable(weight_read_init,  name='weight_read_init')
		self.weight_write_init = tf.Variable(weight_write_init, name='weight_write_init')


	@property
	def state_size(self):
		return [tf.TensorShape((self.n_memory_slots, self.n_memory_units)), 
				tf.TensorShape((self.n_memory_units * self.n_read_heads)),
				tf.TensorShape((self.n_read_heads,  self.n_memory_slots,)),
				tf.TensorShape((self.n_write_heads, self.n_memory_slots,)),
				self.ctrl.state_size]


	@property
	def output_size(self):
		return [tf.TensorShape((self.n_output,))]


	def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
		return [tf.repeat(self.Memory_init, batch_size, axis=0),                      # Memory
				tf.repeat(tf.nn.tanh(self.read_init), batch_size, axis=0),            # read
				tf.repeat(tf.nn.softmax(self.weight_read_init), batch_size, axis=0),  # w_read
				tf.repeat(tf.nn.softmax(self.weight_write_init), batch_size, axis=0), # w_write
				self.ctrl.get_initial_state(inputs, batch_size, dtype)]               # ctrl


	@tf.function
	def call(self, 
			 X, 
			 S, 
			 full=False, 
			 training=False):
		M, R, W_r, W_w, C = S
		Z, C   = self.ctrl(tf.concat([X, R], axis=1), C, training=training)
		Z      = tf.clip_by_value(Z, -self.clip_value, self.clip_value)
		R, W_r = self.read(Z,  W_r, M)
		if full:
			M, W_w, E, A = self.write(Z, W_w, M, full=True)
		else:
			M, W_w = self.write(Z, W_w, M)
		Y      = self.out(tf.concat([R, Z], axis=1))
		Y      = tf.clip_by_value(Y, -self.clip_value, self.clip_value)
		if full:
			return Y, E, A, [M, R, W_r, W_w, C]
		else:
			return Y, [M, R, W_r, W_w, C]


	@tf.function
	def read(self, 
			 Z,
			 W_old, 
			 M,
			 training=False):
		W_new, Rs = [], []
		for head, W in zip(self.read_heads, tf.unstack(W_old, axis=1)):
			R, W = head(Z, W, M)
			W_new.append(W)
			Rs.append(R)
		return tf.concat(Rs, axis=1), tf.stack(W_new, axis=1)


	@tf.function
	def write(self,
			  Z,
			  W_old,
			  M,
			  full=False, 
			  training=False):
		W_new = []
		Es, As, Ws = [], [], []
		for head, W in zip(self.write_heads, tf.unstack(W_old, axis=1)):
			W, A, E = head(Z, W, M)
			Es.append(E)
			As.append(A)
			Ws.append(W)
			W_new.append(W[...,0])
		for E, W in zip(Es, Ws):
			M = M * (1 - tf.matmul(W, E)) 
		for A, W in zip(As, Ws):
			M = M + tf.matmul(W, A)
		if full:
			return M, tf.stack(W_new, axis=1), Es, As
		return M, tf.stack(W_new, axis=1)



class NeuralTuringMachine(tf.keras.Model):
	def __init__(self, 
				 ctrl_class=tf.keras.layers.LSTMCell, 
				 n_ctrl_layers=1, 
				 n_ctrl_units=100, 
				 n_output_units=8, 
				 n_memory_slots=128, 
				 n_memory_units=20, 
				 n_read_heads=1, 
				 n_write_heads=1, 
				 n_shifts=1, 
				 clip_value=20, 
				 global_clipnorm=50, 
				 **kwargs):
		super().__init__(**kwargs)
		self.cell = NeuralTuringMachineCell(ctrl_class, n_ctrl_layers, n_ctrl_units, 
											n_output_units, n_memory_slots, n_memory_units, 
											n_read_heads, n_write_heads, n_shifts, clip_value)
		self.RNN         = tf.keras.layers.RNN(self.cell, return_sequences=True)
		self.optimizer   = tf.keras.optimizers.Adam(learning_rate=1e-3, global_clipnorm=global_clipnorm)
		self.loss        = tf.keras.losses.BinaryCrossentropy()
		current_time     = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
		self.log_writer  = tf.summary.create_file_writer(f"logs/NTM_{current_time}")
		self.loss_metric = tf.keras.metrics.Mean(name="loss")


	@tf.function
	def call(self, X):
		return self.RNN(X)


	def train(self, dataset, epochs):
		metrics = []
		with tqdm(total=epochs) as bar:
			bar.set_description('TRAINING')
			for epoch, batch in zip(range(epochs), dataset):
				X, T, mask = batch
				metric = self.step(X, T, mask)
				metrics.append(metric)
				with self.log_writer.as_default():
					tf.summary.scalar('loss', metric, step=epoch)
				self.loss_metric.reset_state()
				bar.update(1)
		return metrics


	def test(self, dataset, epochs):
		metrics = []
		with tqdm(total=epochs) as bar:
			bar.set_description('Testing')
			for epoch, batch in zip(range(epochs), dataset):
				X, T, mask = batch
				metric = self.target(X, T, mask)
				metrics.append(metric)
				bar.update(1)
		return np.mean(np.array(metrics))


	@tf.function
	def step(self, X, T, mask):
		with tf.GradientTape() as tape:
			L = self.target(X, T, mask)
		gradient = tape.gradient(L, self.trainable_weights)
		self.optimizer.apply_gradients(zip(gradient, self.trainable_weights))
		self.loss_metric.update_state(L)
		return self.loss_metric.result()


	@tf.function
	def target(self, X, T, mask):
		Y = self(X)
		L = self.loss(T, Y, sample_weight=mask)
		return L


	def visualize(self, dataset):
		X, T, mask = next(iter(dataset))
		states, Ys, Es, As = [], [], [], []
		state = self.cell.get_initial_state(batch_size=X.shape[0], dtype=tf.float32)
		#print(state)
		for n in range(X.shape[1]):
			x = X[:,n,:]
			y, e, a, state = self.cell(x, state, full=True)
			states.append(state)
			Ys.append(y)
			Es.append(tf.concat(e, axis=1))
			As.append(tf.concat(a, axis=1))
		Memory, read, weights_read, weights_write, ctrl_state = tuple(zip(*states))
		#Memory        = tf.stack(Memory)
		read          = tf.stack(read)
		weights_read  = tf.stack(weights_read)
		weights_write = tf.stack(weights_write)
		ctrl_state    = tf.stack(ctrl_state)
		Y             = tf.stack(Ys)
		Es            = tf.stack(Es)
		As            = tf.stack(As)
		l             = int(tf.reduce_sum(tf.cast(mask[0,...] != 0, tf.float32)))
		self.display(tf.transpose(X[0,...]),                 'Input')
		self.display(tf.transpose(Y[:,0,...]),               'Output')
		self.display(tf.transpose(T[0,...]),                 'Target')
		self.display(tf.transpose(Y[l+1:2*l+1,0,...]),       'Output cropped')
		self.display(tf.transpose(T[0,l+1:2*l+1,...]),       'Target cropped')
		self.display((tf.transpose(Y[l+1:2*l+1,0,...]) - tf.transpose(T[0,l+1:2*l+1,...]))**2,
															 'Error')
		self.display(tf.transpose(Es[0,:,0,...]),            'erase')
		self.display(tf.transpose(As[0,:,0,...]),            'add')
		self.display(tf.transpose(read[:,0,...]),            'read')
		self.display(tf.transpose(weights_read[:,0,0,...]),  'weights_read')
		self.display(tf.transpose(weights_write[:,0,0,...]), 'weights_write')
		for i, M in enumerate(Memory):
			self.display(tf.transpose(M[0,...]), f'Memory {i}')



	def display(self, X, name):
		plt.title(name)
		plt.imshow(X, cmap='inferno')
		plt.show()




if __name__ == '__main__':
	from datasets import copy_dataset
	with tf.device('GPU'):
		model = NeuralTuringMachine()
		X, Y, mask = next(iter(copy_dataset))
		model(X)
		print(model.test(copy_dataset, 10))
		print(model.train(copy_dataset, 100))
		print(model.test(copy_dataset, 10))
		print(model.summary())
		model.display(copy_dataset)
