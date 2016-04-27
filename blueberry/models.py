# models.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

"""
This file defines a MxNet model, and all code related to training or predicting
using a MxNet model.
"""

import logging
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

class GeneratorIter(DataIter):
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
	def __init__(self, sequences, dnases, contacts, poscontacts, midcontacts, regions, window, data_shapes={}, label_shapes={}, batch_size=1, max_iterations=100):
		super(GeneratorIter, self).__init__()

		self.sequence    = sequences
		self.dnases      = dnases
		self.contacts    = contacts
		self.poscontacts = poscontacts
		self.midcontacts = midcontacts
		self.regions     = regions
		self.n = len(sequences)

		self.window = window
		self.data_shapes = data_shapes
		self.label_shapes = label_shapes
		self.batch_size = batch_size
		self.max_iterations = max_iterations

	@property
	def provide_data(self):
		"""The name and shape of data provided by this iterator"""
		return [(k, tuple([self.batch_size] + list(v))) for k, v in self.data_shapes.items()]

	@property
	def provide_label(self):
		"""The name and shape of label provided by this iterator"""
		return [(k, tuple([self.batch_size] + list(v))) for k, v in self.label_shapes.items()]

	def __iter__(self):
		sequence = self.sequence
		dnases = self.dnases
		contacts = self.contacts
		regions = self.regions
		window = self.window
		batch_size = self.batch_size

		for _iter in range(self.max_iterations):
			data = { 'x1seq' : numpy.zeros((batch_size, 1001, 4)),
			         'x2seq' : numpy.zeros((batch_size, 1001, 4)),
			         'x1dnase' : numpy.zeros((batch_size, 1001, 1)),
			         'x2dnase' : numpy.zeros((batch_size, 1001, 1)) }

			labels = { 'softmax_label' : numpy.zeros(batch_size) }

			i = 0
			while i < batch_size:
				c = numpy.random.randint(self.n)

				if i % 2 == 0:
					k = numpy.random.randint(len(contacts[c]))
					mid1, mid2 = contacts[c][k]
					mid1, mid2 = min(mid1, mid2), max(mid1, mid2)

					if not (LOW_FITHIC_CUTOFF <= mid2 - mid1 <= HIGH_FITHIC_CUTOFF):
						continue 

				else:
					mid1, mid2 = numpy.random.choice(regions[c], 2)
					mid1, mid2 = min(mid1, mid2), max(mid1, mid2)

					if not (LOW_FITHIC_CUTOFF <= mid2 - mid1 <= HIGH_FITHIC_CUTOFF):
						continue

					if self.midcontacts[c].has_key( (mid1, mid2) ):
						continue

					if self.poscontacts[c].has_key( (mid1, mid2) ):
						continue

				labels['softmax_label'][i] = (i+1)%2

				data['x1seq'][i] = sequence[c][mid1-window:mid1+window+1]
				data['x2seq'][i] = sequence[c][mid2-window:mid2+window+1]

				data['x1dnase'][i, :, 0] = dnases[c][mid1-window:mid1+window+1]
				data['x2dnase'][i, :, 0] = dnases[c][mid2-window:mid2+window+1]
				i += 1

			data['x1seq'] = data['x1seq'].reshape(batch_size, 1, 1001, 4)
			data['x2seq'] = data['x2seq'].reshape(batch_size, 1, 1001, 4)
			data['x1dnase'] = data['x1dnase'].reshape(batch_size, 1, 1001, 1)
			data['x2dnase'] = data['x2dnase'].reshape(batch_size, 1, 1001, 1)

			data = [ array(data[l]) for l in self.data_shapes.keys() ]
			labels = [ array(labels[l]) for l in self.label_shapes.keys() ]

			yield DataBatch(data=data, label=labels, pad=0, index=None)

	def reset(self):
		pass

class MxNetModel(object):
	"""A trained mxnet model.

	This will load up a trained MxNet model and for model prediction. Each
	prediction step will load up a saved copy of the model and use it to make
	predictions. This is to allow multiple processes to run in parallel.

	Parameters
	----------
	name : str, optional
		The name of the model file to load up, without .json or .symbol.
		Defaults to 'rambutan-model'

	iteration : int, optional
		The iteration of the model to load for training. Defaults to 1.
	"""

	def __init__( self, name='rambutan-model', iteration=1 ):
		self.name = name
		self.iteration = iteration
	
	def predict( self, chromosome, dnase, contacts, ctx=2 ):
		"""Load up a copy of the model and make predictions.

		A new copy of the model is loaded up each time, so that a single object
		can be used with joblib in order to utilize multiple GPUs are the same
		time.

		Parameters
		----------
		chromosome : array-like, shape (n_nucleotides, 4)
			This is a one-hot encoded matrix of an entire chromosome.

		dnase : array-like, shape (n_nucleotides,)
			This is the raw DNase value for the chromosome, which is a
			positive float between 0 and 200.

		contacts : array-like, shape (n_contacts, 2)
			This matrix specifies contacts between two positions in the genome,
			which are then used to extract regions from the chromosome and dnase.

		ctx : int, optional
			The GPU ID to use. Defaults to 2.
		"""

		n = contacts.shape[0]
		model = mx.model.FeedForward.load( self.name, self.iteration, ctx=mx.gpu(ctx) )
		regions = numpy.union1d( contacts[:,0], contacts[:,1] ).astype('int32')

		region_map = { region: i for i, region in enumerate(regions) }
		region_seq, region_dnase = extract_regions( regions, chromosome, dnase, 500 )

		x1seq = numpy.zeros((n, 1001, 4), dtype='int8')
		x2seq = numpy.zeros((n, 1001, 4), dtype='int8')
		x1dnase = numpy.zeros((n, 1001, 1), dtype='float32')
		x2dnase = numpy.zeros((n, 1001, 1), dtype='float32')
		y = numpy.zeros((n,), dtype='float32')

		for i in range(n):
			mid1, mid2 = region_map[contacts[i, 0]], region_map[contacts[i, 1]]

			x1seq[i] = region_seq[mid1]
			x2seq[i] = region_seq[mid2]

			x1dnase[i] = region_dnase[mid1]
			x2dnase[i] = region_dnase[mid2]

		X = mx.io.NDArrayIter({  
							  'x1seq'   : x1seq.reshape(n, 1, 1001, 4),
							  'x2seq'   : x2seq.reshape(n, 1, 1001, 4),
							  'x1dnase' : x1dnase.reshape(n, 1, 1001, 1),
							  'x2dnase' : x2dnase.reshape(n, 1, 1001, 1)
						  }, batch_size=1024 if 1024 < n else n)
			
		return model.predict(X)[:, 1]

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
	conv1 = Convolution( seq, 96, (5, 4), pad=(2, 0) )
	pool1 = Pooling( conv1, kernel=(3, 1), stride=(3, 1), pool_type='max' )
	conv2 = Convolution( pool1, 128, (5, 1), pad=(2, 0) )
	pool2 = Pooling( conv2, kernel=(3, 1), stride=(3, 1), pool_type='max' )
	return pool2

def DNase( dnase ):
	pool = Pooling( dnase, kernel=(9, 1), stride=(9, 1), pool_type='avg' )
	return pool

def Rambutan(**kwargs):
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
	#x1coord = Variable( name="x1coord" )

	x2seq = Variable( name="x2seq" )
	x2dnase = Variable( name="x2dnase" )
	#x2coord = Variable( name="x2coord" )

	x1 = Concat( Seq(x1seq), DNase(x1dnase) )
	x1_conv1 = Convolution( x1, 164, (1, 1) )
	x1_conv2 = Convolution( x1_conv1, 164, (3, 1) )
	x1_pool = Flatten( Pooling( x1_conv2, kernel=(1001, 1), stride=(1001, 1), pool_type='max' ) )
	x1_ip1 = Dense( x1_pool, 256 )

	x2 = Concat( Seq(x2seq), DNase(x2dnase) )
	x2_conv1 = Convolution( x2, 164, (1, 1) )
	x2_conv2 = Convolution( x2_conv1, 164, (3, 1) )
	x2_pool = Flatten( Pooling( x2_conv2, kernel=(1001, 1), stride=(1001, 1), pool_type='max' ) )
	x2_ip1 = Dense( x2_pool, 256 )

	x = Concat( x1_ip1, x2_ip1 )
	#x = mx.symbol.abs( x1coord - x2coord )
	x_ip1 = Dense( x, 512 )
	x_ip2 = Dense( x_ip1, 512 )
 
	y_p = mx.symbol.FullyConnected( x_ip2, num_hidden=2 )
	softmax = SoftmaxOutput( data=y_p, name='softmax' )
	model = mx.model.FeedForward( symbol=softmax, **kwargs )
	return model