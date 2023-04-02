import tensorflow as tf
from random import randint
import numpy as np



def preprocess(data):
	data = data.batch(32)
	#data = data.cache()
	data = data.prefetch(4)
	return data



def bitseq(n_bits):
	return list(map(int, bin(randint(1, 254))[2:].zfill(n_bits)))



def copy_generator():
	n_bits = 8
	max_length = 10
	null = [[0.0] * n_bits]
	full = [[1.0] * n_bits]
	while True:
		length = max_length
		l = [bitseq(n_bits) for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))
		mask = tf.constant([[0.0]] * (length + 1) + [[1.0]] * length + [[0.0]] * 2 * (max_length - length), dtype=tf.float32)
		yield x, y, mask



def copy_scaled_generator():
	n_bits = 8
	max_length = 10
	null = [[0.0] * n_bits]
	full = [[1.0] * n_bits]
	while True:
		length = max_length
		l = [bitseq(n_bits) for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))
		mask = tf.constant([[0.0]] * (length + 1) + [[1.0/length]] * length + [[0.0]] * 2 * (max_length - length), dtype=tf.float32)
		yield x, y, mask



def copy_token_generator():
	n_bits = 8
	max_length = 10
	null = [[0.0] * (n_bits + 1)]
	full = [[1.0] * (n_bits + 1)]
	full = [[0.0] * n_bits + [1.0]]
	while True:
		length = max_length
		l = [bitseq(n_bits) + [0.0] for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))[:,:n_bits,...]
		mask = tf.constant([[0.0]] * (length + 1) + [[1.0]] * length + [[0.0]] * 2 * (max_length - length), dtype=tf.float32)
		yield x, y, mask



def copy_token_scaled_generator():
	n_bits = 8
	max_length = 10
	null = [[0.0] * (n_bits + 1)]
	full = [[1.0] * (n_bits + 1)]
	full = [[0.0] * n_bits + [1.0]]
	while True:
		length = max_length
		l = [bitseq(n_bits) + [0.0] for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))[:,:n_bits,...]
		mask = tf.constant([[0.0]] * (length + 1) + [[1.0/length]] * length + [[0.0]] * 2 * (max_length - length), dtype=tf.float32)
		yield x, y, mask



def copy_random_generator():
	n_bits = 8
	max_length = 10
	null = [[0.0] * n_bits]
	full = [[1.0] * n_bits]
	while True:
		length = randint(1, max_length)
		l = [bitseq(n_bits) for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))
		mask = tf.constant([[0.0]] * (length + 1) + [[1.0]] * length + [[0.0]] * 2 * (max_length - length), dtype=tf.float32)
		yield x, y, mask



def copy_random_scaled_generator():
	n_bits = 8
	max_length = 10
	null = [[0.0] * n_bits]
	full = [[1.0] * n_bits]
	while True:
		length = randint(1, max_length)
		l = [bitseq(n_bits) for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))
		mask = tf.constant([[0.0]] * (length + 1) + [[1.0/length]] * length + [[0.0]] * 2 * (max_length - length), dtype=tf.float32)
		yield x, y, mask



def copy_random_token_generator():
	n_bits = 8
	max_length = 10
	null = [[0.0] * (n_bits + 1)]
	full = [[1.0] * (n_bits + 1)]
	full = [[0.0] * n_bits + [1.0]]
	while True:
		length = randint(1, max_length)
		l = [bitseq(n_bits) + [0.0] for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))[:,:n_bits,...]
		mask = tf.constant([[0.0]] * (length + 1) + [[1.0]] * length + [[0.0]] * 2 * (max_length - length), dtype=tf.float32)
		yield x, y, mask



def copy_random_token_scaled_generator():
	n_bits = 8
	max_length = 10
	null = [[0.0] * (n_bits + 1)]
	full = [[1.0] * (n_bits + 1)]
	full = [[0.0] * n_bits + [1.0]]
	while True:
		length = randint(1, max_length)
		l = [bitseq(n_bits) + [0.0] for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))[:,:n_bits,...]
		mask = tf.constant([[0.0]] * (length + 1) + [[1.0/length]] * length + [[0.0]] * 2 * (max_length - length), dtype=tf.float32)
		yield x, y, mask



def copy_random_nomask_generator():
	n_bits = 8
	max_length = 10
	null = [[0.0] * n_bits]
	full = [[1.0] * n_bits]
	while True:
		length = randint(1, max_length)
		l = [bitseq(n_bits) for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))
		mask = tf.constant([[1.0]] * (2 * max_length + 1), dtype=tf.float32)
		yield x, y, mask



def copy_random_nomask_scaled_generator():
	n_bits = 8
	max_length = 10
	null = [[0.0] * n_bits]
	full = [[1.0] * n_bits]
	while True:
		length = randint(1, max_length)
		l = [bitseq(n_bits) for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))
		mask = tf.constant([[1.0/length]] * (2 * max_length + 1), dtype=tf.float32)
		yield x, y, mask



def copy_random_nomask_token_generator():
	n_bits = 8
	max_length = 10
	null = [[0.0] * (n_bits + 1)]
	full = [[1.0] * (n_bits + 1)]
	full = [[0.0] * n_bits + [1.0]]
	while True:
		length = randint(1, max_length)
		l = [bitseq(n_bits) + [0.0] for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))[:,:n_bits,...]
		mask = tf.constant([[1.0]] * (2 * max_length + 1), dtype=tf.float32)
		yield x, y, mask



def copy_random_nomask_token_scaled_generator():
	n_bits = 8
	max_length = 10
	null = [[0.0] * (n_bits + 1)]
	full = [[1.0] * (n_bits + 1)]
	full = [[0.0] * n_bits + [1.0]]
	while True:
		length = randint(1, max_length)
		l = [bitseq(n_bits) + [0.0] for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))[:,:n_bits,...]
		mask = tf.constant([[1.0/length]] * (2 * max_length + 1), dtype=tf.float32)
		yield x, y, mask


### EXTENSION ####


def copy_token_generator_100():
	n_bits = 8
	max_length = 100
	null = [[0.0] * (n_bits + 1)]
	full = [[1.0] * (n_bits + 1)]
	full = [[0.0] * n_bits + [1.0]]
	while True:
		length = max_length
		l = [bitseq(n_bits) + [0.0] for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))[:,:n_bits,...]
		mask = tf.constant([[0.0]] * (length + 1) + [[1.0/length]] * length + [[0.0]] * 2 * (max_length - length), dtype=tf.float32)
		yield x, y, mask


def copy_generator_100():
	n_bits = 8
	max_length = 100
	null = [[0.0] * n_bits]
	full = [[1.0] * n_bits]
	while True:
		length = max_length
		l = [bitseq(n_bits) for _ in range(length)]
		x = tf.constant(l + full + null * (max_length * 2 - length), dtype=tf.float32)
		y = tf.constant(null * (length + 1) + l + null * 2 * (max_length - length))
		mask = tf.constant([[0.0]] * (length + 1) + [[1.0]] * length + [[0.0]] * 2 * (max_length - length), dtype=tf.float32)
		yield x, y, mask



copy_dataset = tf.data.Dataset.from_generator(copy_generator, output_signature=(
	tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_dataset = copy_dataset.apply(preprocess)

copy_scaled_dataset = tf.data.Dataset.from_generator(copy_scaled_generator, output_signature=(
	tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_scaled_dataset = copy_scaled_dataset.apply(preprocess)

copy_token_dataset = tf.data.Dataset.from_generator(copy_token_generator, output_signature=(
	tf.TensorSpec(shape=(None, 9)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_token_dataset = copy_token_dataset.apply(preprocess)

copy_token_scaled_dataset = tf.data.Dataset.from_generator(copy_token_scaled_generator, output_signature=(
	tf.TensorSpec(shape=(None, 9)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_token_scaled_dataset = copy_token_scaled_dataset.apply(preprocess)

copy_random_dataset = tf.data.Dataset.from_generator(copy_random_generator, output_signature=(
	tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_random_dataset = copy_random_dataset.apply(preprocess)

copy_random_scaled_dataset = tf.data.Dataset.from_generator(copy_random_scaled_generator, output_signature=(
	tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_random_scaled_dataset = copy_random_scaled_dataset.apply(preprocess)

copy_random_token_dataset = tf.data.Dataset.from_generator(copy_random_token_generator, output_signature=(
	tf.TensorSpec(shape=(None, 9)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_random_token_dataset = copy_random_token_dataset.apply(preprocess)

copy_random_token_scaled_dataset = tf.data.Dataset.from_generator(copy_random_token_scaled_generator, output_signature=(
	tf.TensorSpec(shape=(None, 9)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_random_token_scaled_dataset = copy_random_token_scaled_dataset.apply(preprocess)

copy_random_nomask_dataset = tf.data.Dataset.from_generator(copy_random_nomask_generator, output_signature=(
	tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_random_nomask_dataset = copy_random_nomask_dataset.apply(preprocess)

copy_random_nomask_scaled_dataset = tf.data.Dataset.from_generator(copy_random_nomask_scaled_generator, output_signature=(
	tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_random_nomask_scaled_dataset = copy_random_nomask_scaled_dataset.apply(preprocess)

copy_random_nomask_token_dataset = tf.data.Dataset.from_generator(copy_random_nomask_token_generator, output_signature=(
	tf.TensorSpec(shape=(None, 9)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_random_nomask_token_dataset = copy_random_nomask_token_dataset.apply(preprocess)

copy_random_nomask_token_scaled_dataset = tf.data.Dataset.from_generator(copy_random_nomask_token_scaled_generator, output_signature=(
	tf.TensorSpec(shape=(None, 9)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_random_nomask_token_scaled_dataset = copy_random_nomask_token_scaled_dataset.apply(preprocess)


### EXTENSION ###

copy_dataset_100 = tf.data.Dataset.from_generator(copy_generator_100, output_signature=(
	tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_dataset_100 = copy_dataset_100.apply(preprocess)

copy_token_dataset_100 = tf.data.Dataset.from_generator(copy_token_generator_100, output_signature=(
	tf.TensorSpec(shape=(None, 9)), tf.TensorSpec(shape=(None, 8)), tf.TensorSpec(shape=(None, 1))))
copy_token_dataset_100 = copy_token_dataset_100.apply(preprocess)

if __name__ == '__main__':
	print(next(iter(copy_dataset)))