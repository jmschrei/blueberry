# models.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

"""
This file defines a MxNet model, and all code related to training or predicting
using a MxNet model.
"""

import logging, time
import numpy, os, pyximport

os.environ['CFLAGS'] = ' -I' + numpy.get_include()
pyximport.install()

from blueberry import *
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

class DecimationGenerator(DataIter):
	"""Generator iterator, collects batches from a generator showing a full subset.

	Use on only one chromosome for now."""

	def __init__(self, sequence, dnase, contacts, regions, window, data_shapes={}, use_coord=False, batch_size=1):
		super(DecimationGenerator, self).__init__()

		self.sequence    = sequence
		self.dnase       = dnase
		self.contacts    = contacts
		self.poscontacts = contacts_to_hashmap(contacts)
		self.regions     = regions
		self.coord       = use_coord

		self.window = window
		self.data_shapes = data_shapes
		self.batch_size = batch_size

	@property
	def provide_data(self):
		"""The name and shape of data provided by this iterator"""
		return [(k, tuple([self.batch_size] + list(v))) for k, v in self.data_shapes.items()]

	@property
	def provide_label(self):
		"""The name and shape of label provided by this iterator"""
		return [('softmax_label', (self.batch_size,))]

	def __iter__(self):
		window = self.window
		batch_size = self.batch_size
		width = window/2
		j = 0

		while j < self.contacts.shape[0]:
			data = { 'x1seq' : numpy.zeros((batch_size, window, 4)),
			         'x2seq' : numpy.zeros((batch_size, window, 4)),
			         'x1dnase' : numpy.zeros((batch_size, window, 1)),
			         'x2dnase' : numpy.zeros((batch_size, window, 1)) }

			if self.coord:
				data['x1coord'] = numpy.zeros((batch_size, 1))
				data['x2coord'] = numpy.zeros((batch_size, 1))

			labels = { 'softmax_label' : numpy.zeros(batch_size) }

			i, y = 0, 0
			while i <= batch_size - 25:
				if j % 500 == 0:
					print j, self.contacts.shape[0]

				if y == 0:
					try:
						mid1, mid2 = self.contacts[j]
					except:
						j = self.contacts.shape[0]
						break

					j += 1
					if not (LOW_FITHIC_CUTOFF <= mid2 - mid1 <= HIGH_FITHIC_CUTOFF):
						continue

					y = 1

				else:
					mid1, mid2 = negative_coordinate_pair(self.regions, self.poscontacts)
					y = 0

				for k in range(-2, 3):
					for l in range(-2, 3):
						m1, m2 = mid1 + k*1000, mid2 + l*1000

						data['x1seq'][i] = self.sequence[m1-width:m1+width]
						data['x2seq'][i] = self.sequence[m2-width:m2+width]

						data['x1dnase'][i, :, 0] = self.dnase[m1-width:m1+width]
						data['x2dnase'][i, :, 0] = self.dnase[m2-width:m2+width]

						if self.coord:
							data['x1coord'][i] = m1
							data['x2coord'][i] = m2

						labels['softmax_label'][i] = y
						i += 1

			data['x1seq'] = data['x1seq'].reshape(batch_size, 1, window, 4)
			data['x2seq'] = data['x2seq'].reshape(batch_size, 1, window, 4)
			data['x1dnase'] = data['x1dnase'].reshape(batch_size, 1, window, 1)
			data['x2dnase'] = data['x2dnase'].reshape(batch_size, 1, window, 1)

			data['x1dnase'][ data['x1dnase'] == 0 ] = 1
			data['x2dnase'][ data['x2dnase'] == 0 ] = 1
			data['x1dnase'] = numpy.log(data['x1dnase'])
			data['x2dnase'] = numpy.log(data['x2dnase'])

			data = [ array(data[l]) for l in self.data_shapes.keys() ]
			labels = [ array(labels['softmax_label']) ]
			yield DataBatch(data=data, label=labels, pad=0, index=None)

	def reset(self):
		pass


class ValidationGenerator(DataIter):
	"""Generator iterator, collects batches from a generator showing a full subset.

	Use on only one chromosome for now."""

	def __init__(self, sequence, dnase, contacts, regions, window, data_shapes={}, use_coord=False, batch_size=1):
		super(ValidationGenerator, self).__init__()

		self.sequence    = sequence
		self.dnase       = dnase
		self.contacts    = contacts
		self.poscontacts = contacts_to_hashmap(contacts)
		self.regions     = regions
		self.coord       = use_coord

		self.window = window
		self.data_shapes = data_shapes
		self.batch_size = batch_size

	@property
	def provide_data(self):
		"""The name and shape of data provided by this iterator"""
		return [(k, tuple([self.batch_size] + list(v))) for k, v in self.data_shapes.items()]

	@property
	def provide_label(self):
		"""The name and shape of label provided by this iterator"""
		return [('softmax_label', (self.batch_size,))]

	def __iter__(self):
		window = self.window
		batch_size = self.batch_size
		width = window/2
		j = 0

		while j < self.contacts.shape[0] - batch_size*2:
			data = { 'x1seq' : numpy.zeros((batch_size, window, 4)),
			         'x2seq' : numpy.zeros((batch_size, window, 4)),
			         'x1dnase' : numpy.zeros((batch_size, window, 1)),
			         'x2dnase' : numpy.zeros((batch_size, window, 1)) }

			if self.coord:
				data['x1coord'] = numpy.zeros((batch_size, 1))
				data['x2coord'] = numpy.zeros((batch_size, 1))

			labels = { 'softmax_label' : numpy.zeros(batch_size) }

			i = 0
			while i < batch_size:
				if i % 2 == 0:
					mid1, mid2 = self.contacts[j]
					j += 1
					if not (LOW_FITHIC_CUTOFF <= mid2 - mid1 <= HIGH_FITHIC_CUTOFF):
						continue

				else:
					mid1, mid2 = negative_coordinate_pair(self.regions, self.poscontacts)

				labels['softmax_label'][i] = (i+1)%2

				data['x1seq'][i] = self.sequence[mid1-width:mid1+width]
				data['x2seq'][i] = self.sequence[mid2-width:mid2+width]

				data['x1dnase'][i, :, 0] = self.dnase[mid1-width:mid1+width]
				data['x2dnase'][i, :, 0] = self.dnase[mid2-width:mid2+width]

				if self.coord:
					data['x1coord'][i] = mid1
					data['x2coord'][i] = mid2

				i += 1

			data['x1seq'] = data['x1seq'].reshape(batch_size, 1, window, 4)
			data['x2seq'] = data['x2seq'].reshape(batch_size, 1, window, 4)
			data['x1dnase'] = data['x1dnase'].reshape(batch_size, 1, window, 1)
			data['x2dnase'] = data['x2dnase'].reshape(batch_size, 1, window, 1)

			data['x1dnase'][ data['x1dnase'] == 0 ] = 1
			data['x2dnase'][ data['x2dnase'] == 0 ] = 1
			data['x1dnase'] = numpy.log(data['x1dnase'])
			data['x2dnase'] = numpy.log(data['x2dnase'])

			data = [ array(data[l]) for l in self.data_shapes.keys() ]
			labels = [ array(labels['softmax_label']) ]

			yield DataBatch(data=data, label=labels, pad=0, index=None)

	def reset(self):
		pass


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
	def __init__(self, sequences, dnases, contacts, regions, window, data_shapes={}, use_coord=False, batch_size=1):
		super(TrainingGenerator, self).__init__()

		self.sequence     = sequences
		self.dnases       = dnases
		self.contacts     = contacts
		self.poscontacts  = [ contacts_to_hashmap(contacts) for contacts in self.contacts ]
		self.regions      = regions
		self.coord        = use_coord
		self.n = len(sequences)

		self.window = window
		self.data_shapes = data_shapes
		self.batch_size = batch_size

	@property
	def provide_data(self):
		"""The name and shape of data provided by this iterator"""
		return [(k, tuple([self.batch_size] + list(v))) for k, v in self.data_shapes.items()]

	@property
	def provide_label(self):
		"""The name and shape of label provided by this iterator"""
		return [('softmax_label', (self.batch_size,))]

	def __iter__(self):
		sequence = self.sequence
		dnases = self.dnases
		contacts = self.contacts
		regions = self.regions
		window = self.window
		batch_size = self.batch_size
		width = window/2

		while True:
			data = { 'x1seq' : numpy.zeros((batch_size, window, 4)),
			         'x2seq' : numpy.zeros((batch_size, window, 4)),
			         'x1dnase' : numpy.zeros((batch_size, window, 1)),
			         'x2dnase' : numpy.zeros((batch_size, window, 1)) }

			if self.coord:
				data['x1coord'] = numpy.zeros((batch_size, 1))
				data['x2coord'] = numpy.zeros((batch_size, 1))

			labels = { 'softmax_label' : numpy.zeros(batch_size) }

			i = 0
			while i < batch_size:
				c = numpy.random.randint(self.n)

				if i % 2 == 0:
					k = numpy.random.randint(len(contacts[c]))
					mid1, mid2 = contacts[c][k]
					if not (LOW_FITHIC_CUTOFF <= mid2 - mid1 <= HIGH_FITHIC_CUTOFF):
						continue

				else:
					mid1, mid2 = negative_coordinate_pair(regions[c], self.poscontacts[c])

				labels['softmax_label'][i] = (i+1)%2

				data['x1seq'][i] = sequence[c][mid1-width:mid1+width]
				data['x2seq'][i] = sequence[c][mid2-width:mid2+width]

				data['x1dnase'][i, :, 0] = dnases[c][mid1-width:mid1+width]
				data['x2dnase'][i, :, 0] = dnases[c][mid2-width:mid2+width]

				if self.coord:
					data['x1coord'][i] = mid1
					data['x2coord'][i] = mid2

				i += 1

			data['x1seq'] = data['x1seq'].reshape(batch_size, 1, window, 4)
			data['x2seq'] = data['x2seq'].reshape(batch_size, 1, window, 4)
			data['x1dnase'] = data['x1dnase'].reshape(batch_size, 1, window, 1)
			data['x2dnase'] = data['x2dnase'].reshape(batch_size, 1, window, 1)

			data['x1dnase'][ data['x1dnase'] == 0 ] = 1
			data['x2dnase'][ data['x2dnase'] == 0 ] = 1
			data['x1dnase'] = numpy.log(data['x1dnase'])
			data['x2dnase'] = numpy.log(data['x2dnase'])

			data = [ array(data[l]) for l in self.data_shapes.keys() ]
			labels = [ array(labels['softmax_label']) ]

			yield DataBatch(data=data, label=labels, pad=0, index=None)

	def reset(self):
		pass

def Convolution( data, num_filter, kernel, stride=(1, 1), pad=(0, 0), weight=None, bias=None, beta=None, gamma=None ):
	"""Create a convolution layer with batch normalization and relu activations."""

	conv = mx.symbol.Convolution( data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
	bn = mx.symbol.BatchNorm( data=conv )
	act = mx.symbol.Activation( data=bn, act_type='relu' )
	return act

def Dense( data, num_hidden ):
	"""Create an inner product layer with ReLU activations."""

	ip = FullyConnected( data=data, num_hidden=num_hidden )
	bn = mx.symbol.BatchNorm( data=ip )
	act = mx.symbol.Activation( data=bn, act_type='relu' )
	return act

def Seq( seq ):
	conv1 = Convolution( seq, 32, (7, 4), pad=(3, 0) )
	pool1 = Pooling( conv1, kernel=(3, 1), stride=(3, 1), pool_type='max' )
	conv2 = Convolution( pool1, 32, (7, 1), pad=(3, 0) )
	pool2 = Pooling( conv2, kernel=(3, 1), stride=(3, 1), pool_type='max' )
	return pool2

def DNase( dnase ):
	pool = Pooling( dnase, kernel=(9, 1), stride=(9, 1), pool_type='avg' )
 	return pool

def Arm( seq, dnase ):
	x = Concat( Seq(seq), DNase(dnase) )

	conv1 = Convolution( x, 48, (1, 1) )
	conv2 = Convolution( conv1, 48, (3, 1) )

	pool = Flatten( Pooling( conv2, kernel=(5000, 1), stride=(5000, 1), pool_type='max' ) )
	ip1 = Dense( pool, 128 )
	return ip1

def Rambutan( use_coord=False, **kwargs):
	"""Create the Rambutan model.

	The current default values are the following:

		ctx=[mx.gpu(2), mx.gpu(3)], 
		symbol=Rambutan(),
		epoch_size=5000,
		num_epoch=50,
		learning_rate=0.01,
		wd=0.0,
		optimizer='adam'

	"""
	x1seq = Variable( name="x1seq" )
	x1dnase = Variable( name="x1dnase" )

	x2seq = Variable( name="x2seq" )
	x2dnase = Variable( name="x2dnase" )

	if use_coord:
		x1coord = Variable( name="x1coord" )
		x2coord = Variable( name="x2coord" ) 

	x1 = Arm(x1seq, x1dnase)
	x2 = Arm(x2seq, x2dnase)
	x = Concat(x1, x2)

	if use_coord:
		x = Concat(x, mx.symbol.abs(x1coord - x2coord))

	ip1 = Dense( x, 164 )
	ip2 = Dense( ip1, 164 )
	y_p = mx.symbol.FullyConnected( ip2, num_hidden=2 )
	softmax = SoftmaxOutput( data=y_p, name='softmax' )
	model = mx.model.FeedForward( symbol=softmax, **kwargs )
	return model