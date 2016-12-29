# models.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

"""
This file defines a MxNet model, and all code related to training or predicting
using a MxNet model.
"""

import logging, time
import numpy, os, pyximport

cimport numpy

from .blueberry import *
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score

try:
	import mxnet as mx
	from mxnet.symbol import Pooling, Variable, Flatten, Concat
	from mxnet.symbol import SoftmaxOutput, FullyConnected, Dropout
	from mxnet.io import *
	from mxnet.ndarray import array
	mx.random.seed(0)
except:
	print "ImportWarning: mxnet not imported"
	DataIter = object

from .utils import *

numpy.random.seed(0)
random.seed(0)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def cross_chromosome_dict(contacts):
	d = {}
	for chromosome, mid1, mid2, q in contacts:
		d[chromosome, mid1, mid2] = q
		d[chromosome, mid2, mid1] = q

	return d

class TrainingGenerator(DataIter):
	"""Generator iterator, collects batches from a generator.

	Parameters
	----------
	data : generator

	batch_size : int
		Batch Size

	last_batch_handle : 'pad', 'discard' or 'roll_over'
		How to handle the last batch

	Note
	----
	This iterator will pad, discard or roll over the last batch if
	the size of data does not match batch_size. Roll over is intended
	for training and can cause problems if used for prediction.
	"""
	def __init__(self, sequences, dnases, contacts, regions, window, 
		batch_size=1024, use_seq=True, use_dnase=True, use_dist=True, 
		min_dist=50000, max_dist=1000000):
		super(TrainingGenerator, self).__init__()

		self.sequence     = sequences
		self.dnases       = dnases
		self.contacts     = contacts
		self.contact_dict = cross_chromosome_dict(contacts)
		self.regions      = regions
		self.n            = len(sequences)
		self.use_seq      = use_seq
		self.use_dnase    = use_dnase
		self.use_dist     = use_dist
		self.min_dist     = min_dist
		self.max_dist     = max_dist

		self.window = window
		self.batch_size = batch_size
		self.data_shapes = {'x1seq' : (1, window, 4), 'x2seq' : (1, window, 4), 
			'x1dnase' : (1, window, 8), 'x2dnase' : (1, window, 8), 'distance' : (191,)}

	@property
	def provide_data(self):
		"""The name and shape of data provided by this iterator"""
		return [(k, tuple([self.batch_size] + list(v))) for k, v in self.data_shapes.items()]

	@property
	def provide_label(self):
		"""The name and shape of label provided by this iterator"""
		return [('softmax_label', (self.batch_size,))]

	def __iter__(self):
		cdef numpy.ndarray sequence = self.sequence
		cdef numpy.ndarray dnases = self.dnases
		cdef numpy.ndarray contacts = self.contacts
		cdef numpy.ndarray regions = self.regions
		cdef numpy.ndarray x1dnase, x2dnase
		cdef int window = self.window, batch_size = self.batch_size
		cdef int i, c, k, mid1, mid2, distance, width = window/2, batch
		cdef dict data, labels, contact_dict = self.contact_dict
		cdef list data_list, label_list

		data = { 'x1seq' : numpy.zeros((batch_size, window, 4)),
				 'x2seq' : numpy.zeros((batch_size, window, 4)),
				 'x1dnase' : numpy.zeros((batch_size, window, 8)),
				 'x2dnase' : numpy.zeros((batch_size, window, 8)),
				 'distance' : numpy.zeros((batch_size, 191))}

		labels = { 'softmax_label' : numpy.zeros(batch_size) }

		while True:
			data['x1seq'] = data['x1seq'].reshape(batch_size, window, 4)
			data['x2seq'] = data['x2seq'].reshape(batch_size, window, 4)
			data['x1dnase'] = data['x1dnase'].reshape(batch_size, window, 8) * 0
			data['x2dnase'] = data['x2dnase'].reshape(batch_size, window, 8) * 0

			i = 0
			while i < batch_size:
				if i % 2 == 0:
					k = numpy.random.randint(len(contacts))
					c, mid1, mid2 = contacts[k, :3]
				else:
					mid1 = numpy.random.choice(self.regions[c])
					mid2 = mid1 + numpy.random.choice((self.max_dist - self.min_dist) / window + 1) * window + self.min_dist
					if mid2 > self.regions[c][-1] or contact_dict.has_key((c, mid1, mid2)):
						continue

				labels['softmax_label'][i] = (i+1)%2

				if self.use_seq:
					data['x1seq'][i] = sequence[c][mid1-width:mid1+width]
					data['x2seq'][i] = sequence[c][mid2-width:mid2+width]

				if self.use_dnase:
					data['x1dnase'][i] = dnases[c][mid1-width:mid1+width]
					data['x2dnase'][i] = dnases[c][mid2-width:mid2+width]

				if self.use_dist:
					distance = mid2 - mid1 - self.min_dist
					for k in range(100):
						data['distance'][i][k] = 0 if distance >= k*1000 else 1
					for k in range(91):
						data['distance'][i][k+100] = 0 if distance >= 100000 + k*10000 else 1

				i += 1

			data['x1seq'] = data['x1seq'].reshape(batch_size, 1, window, 4)
			data['x2seq'] = data['x2seq'].reshape(batch_size, 1, window, 4)
			data['x1dnase'] = data['x1dnase'].reshape(batch_size, 1, window, 8)
			data['x2dnase'] = data['x2dnase'].reshape(batch_size, 1, window, 8)

			data_list = [ array(data[key]) for key in self.data_shapes.keys() ]
			label_list = [ array(labels['softmax_label']) ]
			yield DataBatch(data=data_list, label=label_list, pad=0, index=None)

	def reset(self):
		pass

class ValidationGenerator(DataIter):
	"""Generator iterator, collects batches from a generator showing a full subset.

	Use on only one chromosome for now."""

	def __init__(self, sequence, dnase, contacts, regions, window, 
		batch_size=1024, use_seq=True, use_dnase=True, use_dist=True, 
		min_dist=50000, max_dist=1000000):
		super(ValidationGenerator, self).__init__()

		self.sequence     = sequence
		self.dnase        = dnase
		self.contacts     = contacts
		self.regions      = regions
		self.use_seq      = use_seq
		self.use_dnase    = use_dnase
		self.use_dist     = use_dist
		self.min_dist     = min_dist
		self.max_dist     = max_dist

		self.window = window
		self.batch_size = batch_size
		self.data_shapes = {'x1seq' : (1, window, 4), 'x2seq' : (1, window, 4), 
			'x1dnase' : (1, window, 8), 'x2dnase' : (1, window, 8), 'distance' : (191,)}

	@property
	def provide_data(self):
		"""The name and shape of data provided by this iterator"""
		return [(k, tuple([self.batch_size] + list(v))) for k, v in self.data_shapes.items()]

	@property
	def provide_label(self):
		"""The name and shape of label provided by this iterator"""
		return [('softmax_label', (self.batch_size,))]

	def __iter__(self):
		cdef numpy.ndarray sequence = self.sequence
		cdef numpy.ndarray dnase = self.dnase
		cdef dict data, labels
		cdef int i, j = 0, k, batch_size = self.batch_size, window = self.window, l
		cdef int mid1, mid2, distance, width=window/2, last_mid1, last_mid2
		cdef list data_list, label_list
		cdef str key

		data = { 'x1seq' : numpy.zeros((batch_size, window, 4)),
				 'x2seq' : numpy.zeros((batch_size, window, 4)),
				 'x1dnase' : numpy.zeros((batch_size, window, 8)),
				 'x2dnase' : numpy.zeros((batch_size, window, 8)),
				 'distance' : numpy.zeros((batch_size, 191))
		}

		labels = { 'softmax_label' : numpy.zeros(batch_size) }

		j = 0
		l = self.contacts.shape[0] - batch_size*2
		while j < l:
			data['x1seq'] = data['x1seq'].reshape(batch_size, window, 4)
			data['x2seq'] = data['x2seq'].reshape(batch_size, window, 4)
			data['x1dnase'] = data['x1dnase'].reshape(batch_size, window, 8)
			data['x2dnase'] = data['x2dnase'].reshape(batch_size, window, 8)

			i = 0
			while i < batch_size:
				if i % 2 == 0:
					mid1, mid2 = self.contacts[j]
					j += 1
					if not (self.min_dist <= mid2 - mid1 <= self.max_dist) and j < l:
						continue

				else:
					mid1, mid2 = numpy.random.choice(self.regions, 2)
					mid2 = mid1 + numpy.random.choice((self.max_dist - self.min_dist) / window) * window + self.min_dist
					if mid2 > self.regions[-1]:
						continue

				labels['softmax_label'][i] = (i+1)%2

				if self.use_seq:
					data['x1seq'][i] = sequence[mid1-width:mid1+width]
					data['x2seq'][i] = sequence[mid2-width:mid2+width]

				if self.use_dnase:
					data['x1dnase'][i] = dnase[mid1-width:mid1+width]
					data['x2dnase'][i] = dnase[mid2-width:mid2+width]

				if self.use_dist:
					distance = mid2 - mid1 - self.min_dist
					for k in range(100):
						data['distance'][i][k] = 0 if distance >= k*1000 else 1
					for k in range(91):
						data['distance'][i][k+100] = 0 if distance >= 100000 + k*10000 else 1

				i += 1
				last_mid1 = mid1
				last_mid2 = mid2

			data['x1seq'] = data['x1seq'].reshape(batch_size, 1, window, 4)
			data['x2seq'] = data['x2seq'].reshape(batch_size, 1, window, 4)
			data['x1dnase'] = data['x1dnase'].reshape(batch_size, 1, window, 8)
			data['x2dnase'] = data['x2dnase'].reshape(batch_size, 1, window, 8)

			data_list = [array(data[key][:i]) for key in self.data_shapes.keys()]
			label_list = [array(labels['softmax_label'])]
			yield DataBatch(data=data_list, label=label_list, pad=0, index=None)

	def reset(self):
		pass

def Convolution(x, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type='relu'):
	"""Create a convolution layer with batch normalization and relu activations."""

	x = mx.symbol.Convolution(data=x, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, cudnn_tune='fastest')
	x = mx.symbol.BatchNorm(data=x)
	x = mx.symbol.Activation(data=x, act_type=act_type)
	return x

def Dense(x, num_hidden, act_type='relu'):
	"""Create an inner product layer with ReLU activations."""

	x = FullyConnected(data=x, num_hidden=num_hidden)
	x = mx.symbol.BatchNorm(data=x)
	x = mx.symbol.Activation(data=x, act_type=act_type)
	return x

def Arm(seq, dnase):
	seq = Convolution(seq, 48, (7, 4), pad=(3, 0))
	seq = Pooling(seq, kernel=(3, 1), stride=(3, 1), pool_type='max')
	seq = Convolution(seq, 48, (7, 1), pad=(3, 0))
	seq = Pooling(seq, kernel=(3, 1), stride=(3, 1), pool_type='max')

	dnase = Pooling(dnase, kernel=(9, 1), stride=(9, 1), pool_type='max')
	dnase = Convolution(dnase, 8, (1, 8))

	x = Concat(seq, dnase)
	x = Convolution(x, 64, (3, 1))
	x = Convolution(x, 64, (3, 1))
	x = Flatten(Pooling(x, kernel=(107, 1), stride=(107, 1), pool_type='max'))
	x = Dense(x, 256)
	return x

def RambutanSymbol(**kwargs):
	x1seq = Variable(name="x1seq")
	x1dnase = Variable(name="x1dnase")
	x1 = Arm(x1seq, x1dnase)

	x2seq = Variable(name="x2seq")
	x2dnase = Variable(name="x2dnase")
	x2 = Arm(x2seq, x2dnase)

	xd = Variable(name="distance")
	xd = Dense(xd, 64)

	x = Concat(x1, x2, xd)
	x = Dense(x, 256)
	x = mx.symbol.FullyConnected(x, num_hidden=2)
	y = SoftmaxOutput(data=x, name='softmax')
	model = mx.model.FeedForward(symbol=y, **kwargs)
	return model
