import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

# This function selects the probability distribution over actions
from baselines.common.distributions import make_pdtype
from keras import backend as K

MM_EMBEDDINGS_DIM = 50
MM_MAX_WORD_SIZE = 20
MM_MAX_SENTENCE_SIZE = 200
MM_FEATURES_SIZE = 20000
MM_MAX_VOCAB_SIZE = 5000


# Fully connected layer
def fc_layer(inputs, units, activation_fn=tf.nn.relu, gain=1.0):
	return tf.layers.dense(inputs,
							units=units,
							activation=activation_fn,
							kernel_initializer=tf.orthogonal_initializer(gain))


# LSTM Layer
#def lstm_layer(vocab_size=MM_MAX_VOCAB_SIZE, word_len_limit=MM_MAX_WORD_SIZE, input_length=MM_MAX_SENTENCE_SIZE):
#	return LSTM(Embedding(vocab_size, word_len_limit, input_length=input_length, mask_zero=True), dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
def lstm_layer(em_input, vocab_size=MM_MAX_VOCAB_SIZE, word_len_limit=MM_MAX_WORD_SIZE, input_length=MM_MAX_SENTENCE_SIZE):
	print('em shape', em_input.shape)
	embedding = Embedding(vocab_size, word_len_limit, input_length=input_length, mask_zero=True )(em_input)
	print('emb shape', embedding.shape)
	return LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedding)

# future
#x_train = sequence.pad_sequences(x_train, maxlen=maxlen)   # pre_padding with 0

"""
This object creates the PPO Network architecture
"""

class PPOPolicy(object):
	def __init__(self, sess, ob_space, action_space, nbatch, nsteps, reuse=False):
		# This will use to initialize our kernels
		gain = np.sqrt(2)

		self.tokenizer = Tokenizer(num_words=5000)
		# Based on the action space, will select what probability distribution type
		# we will use to distribute action in our stochastic policy (in our case DiagGaussianPdType
		# aka Diagonal Gaussian, 3D normal distribution)
		self.pdtype = make_pdtype(action_space)

#		print('ob_space:', ob_space)
#		print('ac_space:', action_space.n)
#		height, width = 1, ob_space.n
#		ob_shape = (None, ob_space.n)
		ob_shape = ob_space.shape

		text_shape = ( None, 200 )
#		text_shape = ( 1, 7, 33 )

		# Create the input placeholder
		field_inputs_ = tf.placeholder(tf.float32, (None, ob_space.shape[1]), name="field_input")
		combined_inputs_ = tf.placeholder(tf.float32, (None, ob_space.shape[1] + MM_EMBEDDINGS_DIM*2 ), name="combined_input")
		text_inputs_ = tf.placeholder(tf.float32, text_shape, name="text_input")

		available_moves = tf.placeholder(tf.float32, [None, action_space.n], name="availableActions")

		"""
		Build the model
		Embedding
		LSTM

		3 FC for spatial dependiencies
		1 common FC

		1 FC for policy (actor)
		1 FC for value (critic)

		"""
		with tf.variable_scope('model', reuse=reuse):
			# text reading LSTM
#			lt_layer = lstm_layer()
			text_inputs_keras = tf.keras.layers.Input(tensor=text_inputs_)

			text_out = lstm_layer(text_inputs_keras)

			shape = text_out.get_shape().as_list() [1:]       # a list: [None, 9, 2]
			dim = np.prod(shape)            # dim = prod(9,2) = 18
			print('text_flatten before reshape',text_out.shape)
#			text_flatten = tf.reshape(text_out, [-1, dim])           # -1 means "all"
			text_flatten = tf.reshape(text_out, [1, -1])           # -1 means "all"
#			text_flatten =  tf.reshape(text_out, [-1])

			"""
			# state analyzing layers
			print('ooobc_shape',ob_shape)
			fc_1 = fc_layer(field_inputs_, 700, gain=gain)
			print('ooobc_shape2',ob_shape)
			fc_2 = fc_layer(fc_1, 400, gain=gain)
			print('ooobc_shape3',ob_shape)
			fc_3 = fc_layer(fc_2, 200, gain=gain)
			print('fc_3fc_3fc_3fc_3',fc_3.shape)
#			fc_3 = tf.reshape(fc_3, [-1])           # -1 means "all"
			print('text_flatten shape', text_flatten.shape)

			# Make 1 Flatten to 1D
#			fc_3 = tf.reshape(fc_3, tf.stack([-1, np.prod(shape(fc_3)[1:])]))
			shape = fc_3.get_shape().as_list() [1:]       # a list: [None, 9, 2]
			dim = np.prod(shape)            # dim = prod(9,2) = 18
#			fc_3 = tf.reshape(fc_3, [-1, int(dim)])           # -1 means "all"
#			fc_3 = tf.reshape(fc_3, [1, -1])           # -1 means "all"
			print('fc_3 shape', fc_3.shape)
			"""
			# This returns a tensor
			field_inputs_keras = tf.keras.layers.Input(tensor=field_inputs_)

			# a layer instance is callable on a tensor, and returns a tensor
			fc_1 = tf.keras.layers.Dense(700, activation='relu')(field_inputs_keras)
			fc_2 = tf.keras.layers.Dense(400, activation='relu')(fc_1)
			fc_3 = tf.keras.layers.Dense(300, activation='relu')(fc_2)
#			fc_common = tf.concat([tf.reshape(fc_3, [-1]), text_flatten], -1)
#			fc_common = tf.concat([fc_3, text_out], -1)

#			fc_common = tf.keras.layers.Concatenate(axis=-1)([text_flatten, fc_3])
#			print('fc_common shape', fc_common.shape)
			#dense into 512
#			fc_common = fc_layer(fc_common, 512, gain=gain)

			text_dense = tf.keras.layers.Dense(256, activation='relu')(text_out)
			field_dense = tf.keras.layers.Dense(256, activation='relu')(fc_3)
			shape = text_dense.get_shape().as_list() [1:]       # a list: [None, 9, 2]
			dim = np.prod(shape)            # dim = prod(9,2) = 18
			text_dense = tf.reshape(text_dense, [-1, int(dim)])           # -1 means "all"

#			scaled_image = tf.keras.layers.Lambda(function=lambda tensors: tensors[0] * tensors[1])([image, scale])
#			fc_common_dense = Lambda(lambda x:K.concatenate([x[0], x[1]], axis=1))([text_dense, field_dense])
#			fc_common_dense = tf.keras.layers.Concatenate(axis=-1)(list([text_dense, field_dense]))
			fc_common_dense = tf.keras.layers.Concatenate(axis=-1)(list([text_dense, field_dense]))
			fc_common_dense = tf.keras.layers.Dense(256, activation='relu')(fc_common_dense)


			# This build a fc connected layer that returns a probability distribution
			# over actions (self.pd) and our pi logits (self.pi).
			self.pd, self.pi = self.pdtype.pdfromlatent(fc_common_dense, init_scale=0.01)

			# Calculate the v(s)
#			vf = fc_layer(fc_3, 1, activation_fn=None)[:,0]
			vf = fc_layer(fc_common_dense, 1, activation_fn=None)[:,0]

		self.initial_state = None

		"""
		# Take an action in the action distribution (remember we are in a situation
		# of stochastic policy so we don't always take the action with the highest probability
		# for instance if we have 2 actions 0.7 and 0.3 we have 30% channce to take the second)
		a0 = self.pd.sample()

		# Calculate the neg log of our probability
		neglogp0 = self.pd.neglogp(a0)
		"""

		# perform calculations using available moves lists
		availPi = tf.add(self.pi, available_moves)

		def sample():
			u = tf.random_uniform(tf.shape(availPi))
			return tf.argmax(availPi - tf.log(-tf.log(u)), axis=-1)

		a0 = sample()
		el0in = tf.exp(availPi - tf.reduce_max(availPi, axis=-1, keep_dims=True))
		z0in = tf.reduce_sum(el0in, axis=-1, keep_dims = True)
		p0in = el0in / z0in
		onehot = tf.one_hot(a0, availPi.get_shape().as_list()[-1])
		neglogp0 = -tf.log(tf.reduce_sum(tf.multiply(p0in, onehot), axis=-1))


		# Function use to take a step returns action to take and V(s)
		def step(state_in, valid_moves, ob_texts,*_args, **_kwargs):
			# return a0, vf, neglogp0
			# padd text
#			print('ob_text', ob_texts)
			for ob_text in ob_texts:
#				print('ob_text', ob_text)
				self.tokenizer.fit_on_texts([ob_text])

			ob_text_input = []
			for ob_text in ob_texts:
#				print('ob_text', ob_text)
				token = self.tokenizer.texts_to_sequences([ob_text])
				token = sequence.pad_sequences(token, maxlen=MM_MAX_SENTENCE_SIZE)   # pre_padding with 0
				ob_text_input.append(token)
#				print('token', token)
#				print('token shape', token.shape)
			orig_ob_text_input = np.array(ob_text_input)
			shape = orig_ob_text_input.shape
#			print('ob_text_input shape', shape)
			ob_text_input = orig_ob_text_input.reshape(shape[0], shape[2])

			try:
				return sess.run([a0,vf, neglogp0], {field_inputs_: state_in, text_inputs_:ob_text_input, available_moves:valid_moves})
			except Exception as e:
				print('Issue processing step!!!')
				print('printing data')
				print('valid_moves:',valid_moves)
				print('ob_texts:',ob_texts)
				print('ob_text_input:',ob_text_input)
				print('orig_ob_text_input:',orig_ob_text_input)
				raise e

		# Function that calculates only the V(s)
		def value(state_in, valid_moves, ob_texts, *_args, **_kwargs):
			for ob_text in ob_texts:
#				print('ob_text', ob_text)
				self.tokenizer.fit_on_texts([ob_text])

			ob_text_input = []
			for ob_text in ob_texts:
#				print('ob_text', ob_text)
				token = self.tokenizer.texts_to_sequences([ob_text])
				token = sequence.pad_sequences(token, maxlen=MM_MAX_SENTENCE_SIZE)   # pre_padding with 0
				ob_text_input.append(token)
#				print('token', token)
#				print('token shape', token.shape)
			ob_text_input = np.array(ob_text_input)
			shape = ob_text_input.shape
#			print('ob_text_input shape', shape)
			ob_text_input = ob_text_input.reshape(shape[0], shape[2])
			try:
				return sess.run(vf, {field_inputs_:state_in, text_inputs_:ob_text_input, available_moves:valid_moves})
			except Exception as e:
				print('Issue processing step!!!')
				print('printing data')
				print('valid_moves:',valid_moves)
				print('ob_texts:',ob_texts)
				print('ob_text_input:',ob_text_input)
				raise e

		def select_action(state_in, valid_moves, ob_texts, *_args, **_kwargs):
			for ob_text in ob_texts:
#				print('ob_text', ob_text)
				self.tokenizer.fit_on_texts([ob_text])

			ob_text_input = []
			for ob_text in ob_texts:
#				print('ob_text', ob_text)
				token = self.tokenizer.texts_to_sequences([ob_text])
				token = sequence.pad_sequences(token, maxlen=MM_MAX_SENTENCE_SIZE)   # pre_padding with 0
				ob_text_input.append(token)
#				print('token', token)
#				print('token shape', token.shape)
			ob_text_input = np.array(ob_text_input)
			shape = ob_text_input.shape
#			print('ob_text_input shape', shape)
			ob_text_input = ob_text_input.reshape(shape[0], shape[2])
			try:
				return sess.run(vf, {field_inputs_:state_in, text_inputs_:ob_text_input, available_moves:valid_moves})
			except Exception as e:
				print('Issue processing step!!!')
				print('printing data')
				print('valid_moves:',valid_moves)
				print('ob_texts:',ob_texts)
				print('ob_text_input:',ob_text_input)
				raise e

		self.field_inputs_ = field_inputs_
		self.text_inputs_ = text_inputs_
		self.available_moves = available_moves
		self.vf = vf
		self.step = step
		self.value = value
		self.select_action = select_action
		print('this did finish')
