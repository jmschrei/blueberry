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
from libc.stdlib cimport calloc
from libc.stdlib cimport free

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

class MultiAUC(mx.metric.EvalMetric):
	"""Calculate accuracies of multi label"""

	def __init__(self, num=None):
		super(MultiAUC, self).__init__('multi-accuracy', num)

	def update(self, labels, preds):
		mx.metric.check_label_shapes(labels, preds)

		for i, (y_true, y_pred) in enumerate(zip(labels, preds)):
			y_pred = y_pred.asnumpy()[:,1]
			y_true = y_true.asnumpy().astype('int32')

			y_pred = y_pred[y_true != -1]
			y_true = y_true[y_true != -1]

			if y_true.shape[0] < 2:
				pass
			elif numpy.unique(y_true).shape[0] < 2:
				pass 
			else:
				self.sum_metric[i] += roc_auc_score(y_true, y_pred)
				self.num_inst[i] += 1

def cross_celltype_dict( contacts ):
	"""Take in a contact map and return a dictionary."""

	d = {}
	for celltype, chromosome, mid1, mid2, p in contacts:
		d[celltype, chromosome, mid1, mid2] = p
		d[celltype, chromosome, mid2, mid1] = p

	return d

def cross_chromosome_dict(contacts):
	d = {}
	for chromosome, mid1, mid2, p in contacts:
		d[chromosome, mid1, mid2] = p
		d[chromosome, mid2, mid1] = p

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
		min_dist=25000, max_dist=10000000):
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
			'x1dnase' : (1, window, 8), 'x2dnase' : (1, window, 8), 'distance' : (281,)}

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
				 'distance' : numpy.zeros((batch_size, 281))}

		labels = { 'softmax_label' : numpy.zeros(batch_size) }

		for batch in range(self.n_batches):
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
					mid1, mid2 = numpy.random.choice(regions[c], 2)
					if contact_dict.has_key((c, mid1, mid2)):
						continue

				mid1, mid2 = min(mid1, mid2), max(mid1, mid2)

				if self.use_seq:
					data['x1seq'][i] = sequence[c][mid1-width:mid1+width]
					data['x2seq'][i] = sequence[c][mid2-width:mid2+width]

				if self.use_dnase:
					data['x1dnase'][i] = dnases[c][mid1-width:mid1+width]
					data['x2dnase'][i] = dnases[c][mid2-width:mid2+width]

				if self.use_dist:
					distance = mid2 - mid1 - self.min_dist
					for k in range(100):
						data['distance'][i][k] = 1 if distance >= k*1000 else 0
					for k in range(91):
						data['distance'][i][k+100] = 1 if distance >= 100000 + k*10000 else 0
					for k in range(91):
						data['distance'][i][k+190] = 1 if distance >= 1000000 + k*100000 else 0

				i += 1

			data['x1seq'] = data['x1seq'].reshape(batch_size, 1, window, 4)
			data['x2seq'] = data['x2seq'].reshape(batch_size, 1, window, 4)
			data['x1dnase'] = data['x1dnase'].reshape(batch_size, 1, window, 8)
			data['x2dnase'] = data['x2dnase'].reshape(batch_size, 1, window, 8)

			data_list = [array(data[key]) for key in self.data_shapes.keys()]
			label_list = [array(labels['softmax_label'])]
			yield DataBatch(data=data_list, label=label_list, pad=0, index=None)

		raise StopIteration

	def reset(self):
		pass

class ValidationGenerator(DataIter):
	"""Generator iterator, collects batches from a generator showing a full subset.

	Use on only one chromosome for now."""

	def __init__(self, sequence, dnase, contacts, regions, window, 
		batch_size=1024, use_seq=True, use_dnase=True, use_dist=True, 
		min_dist=25000, max_dist=10000000):
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
			'x1dnase' : (1, window, 8), 'x2dnase' : (1, window, 8), 'distance' : (281,)}

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
		cdef list short_regions = range(25000, 100000, 1000)
		cdef list mid_regions = range(100000, 1000000, 1000)
		cdef list long_regions = range(1000000, 10000000, 1000)

		data = { 'x1seq' : numpy.zeros((batch_size, window, 4)),
				 'x2seq' : numpy.zeros((batch_size, window, 4)),
				 'x1dnase' : numpy.zeros((batch_size, window, 8)),
				 'x2dnase' : numpy.zeros((batch_size, window, 8)),
				 'distance' : numpy.zeros((batch_size, 281))
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
					mid1, mid2 = min(mid1, mid2), max(mid1, mid2)

				labels['softmax_label'] = (i+1)%2

				if self.use_seq:
					data['x1seq'][i] = sequence[mid1-width:mid1+width]
					data['x2seq'][i] = sequence[mid2-width:mid2+width]

				if self.use_dnase:
					data['x1dnase'][i] = dnase[mid1-width:mid1+width]
					data['x2dnase'][i] = dnase[mid2-width:mid2+width]

				if self.use_dist:
					distance = mid2 - mid1 - self.min_dist
					for k in range(100):
						data['distance'][i][k] = 1 if distance >= k*1000 else 0
					for k in range(91):
						data['distance'][i][k+100] = 1 if distance >= 100000 + k*10000 else 0
					for k in range(91):
						data['distance'][i][k+190] = 1 if distance >= 1000000 + k*100000 else 0

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

class MultiCellTypeGenerator(DataIter):
	"""Generator iterator, collects batches from a generator showing a full subset."""

	def __init__(self, sequences, dnases, contacts, regions, window, batch_size=1):
		super(MultiCellTypeGenerator, self).__init__()

		self.sequence      = sequences
		self.dnases        = dnases
		self.contacts      = contacts
		self.contact_dict  = cross_celltype_dict(contacts)
		self.regions       = regions
		self.celltypes     = numpy.unique(contacts[:,0])

		self.m = self.celltypes.shape[0]
		self.n = len(sequences)

		self.window = window
		self.batch_size = batch_size
		self.data_shapes = {'x1seq' : (1, window, 4), 'x2seq' : (1, window, 4), 
			'x1dnase' : (1, window, 8), 'x2dnase' : (1, window, 8), 'distance' : (281,)}

	@property
	def provide_data(self):
		"""The name and shape of data provided by this iterator"""
		return [(k, tuple([self.batch_size] + list(v))) for k, v in self.data_shapes.items()]

	@property
	def provide_label(self):
		"""The name and shape of label provided by this iterator"""
		return [('softmax_label', (self.batch_size,))]

	def __iter__(self):
		cdef numpy.ndarray sequence   = self.sequence
		cdef numpy.ndarray dnases     = self.dnases
		cdef numpy.ndarray contacts   = self.contacts
		cdef numpy.ndarray regions    = self.regions
		cdef int window     = self.window
		cdef int batch_size = self.batch_size
		cdef int width      = int(window/2)
		cdef dict contact_dict = self.contact_dict

		cdef int i, l, c, d, mid1, mid2, distance, k
		cdef dict data, labels
		cdef str key
		cdef list data_list, label_list

		data = { 'x1seq' : numpy.zeros((batch_size, window, 4)),
				 'x2seq' : numpy.zeros((batch_size, window, 4)),
				 'x1dnase' : numpy.zeros((batch_size, window, 8)),
				 'x2dnase' : numpy.zeros((batch_size, window, 8)),
				 'distance' : numpy.zeros((batch_size, 281)) }

		labels = { 'softmax_label' : numpy.zeros(batch_size) }

		while True:
			data['x1seq'] = data['x1seq'].reshape(batch_size, window, 4)
			data['x2seq'] = data['x2seq'].reshape(batch_size, window, 4)
			data['x1dnase'] = data['x1dnase'].reshape(batch_size, window, 8)
			data['x2dnase'] = data['x2dnase'].reshape(batch_size, window, 8)

			l = contacts.shape[0]
			i = 0
			while i < batch_size:
				if i % 2 == 0:
					k = numpy.random.randint(l)
					d, c, mid1, mid2 = contacts[k, :4]
					if not (LOW_FITHIC_CUTOFF <= mid2 - mid1 <= HIGH_FITHIC_CUTOFF):
						continue

				else:
					d = numpy.random.choice(self.celltypes)
					c = numpy.random.randint(self.n)
					if (d == 1 or d == 2) and c == 8:
						continue

					while True:
						mid1, mid2 = numpy.random.choice(regions[d, c], 2)
						mid1, mid2 = min(mid1, mid2), max(mid1, mid2)
						if not contact_dict.has_key( (d, c, mid1, mid2) ):
							break

				labels['softmax_label'][i] = (i+1)%2

				data['x1seq'][i] = sequence[c][mid1-width:mid1+width]
				data['x2seq'][i] = sequence[c][mid2-width:mid2+width]

				data['x1dnase'][i] = dnases[d][c][mid1-width:mid1+width]
				data['x2dnase'][i] = dnases[d][c][mid2-width:mid2+width]

				distance = mid2 - mid1 - LOW_FITHIC_CUTOFF
				for k in range(100):
					data['distance'][i][k] = 1 if distance >= k*1000 else 0
				for k in range(91):
					data['distance'][i][k+100] = 1 if distance >= 100000 + k*10000 else 0
				for k in range(91):
					data['distance'][i][k+190] = 1 if distance >= 1000000 + k*100000 else 0

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

def Rambutan(**kwargs):
	x1seq = Variable(name="x1seq")
	x1dnase = Variable(name="x1dnase")

	x1seq = Convolution(x1seq, 48, (7, 4))
	x1seq = Pooling(x1seq, kernel=(3, 1), stride=(3, 1), pool_type='max')
	x1seq = Convolution(x1seq, 48, (7, 1))
	x1seq = Flatten(Pooling(x1seq, kernel=(325, 1), stride=(325, 1), pool_type='max'))

	x1chrom = Flatten(Pooling(x1dnase, kernel=(1000, 1), stride=(1000, 1), pool_type='avg'))
	x1chrom = Dense(x1chrom, 96)

	x1 = Concat(x1seq, x1chrom)
	x1 = Dense(x1, 256)

	x2seq = Variable(name="x2seq")
	x2dnase = Variable(name="x2dnase")

	x2seq = Convolution(x2seq, 48, (7, 4))
	x2seq = Pooling(x2seq, kernel=(3, 1), stride=(3, 1), pool_type='max')
	x2seq = Convolution(x2seq, 48, (7, 1))
	x2seq = Flatten(Pooling(x2seq, kernel=(325, 1), stride=(325, 1), pool_type='max'))

	x2chrom = Flatten(Pooling(x2dnase, kernel=(1000, 1), stride=(1000, 1), pool_type='avg'))
	x2chrom = Dense(x2chrom, 96)

	x2 = Concat(x2seq, x2chrom)
	x2 = Dense(x2, 256)

	xd = Variable(name="distance")
	xd = Dense(xd, 64)

	x = Concat(x1, x2, xd)
	x = Dense(x, 256)
	x = Dense(x, 256)
	x = mx.symbol.FullyConnected(x, num_hidden=2)
	y = SoftmaxOutput(data=x, name='softmax')
	model = mx.model.FeedForward(symbol=y, **kwargs)
	return model
