# utils.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

"""
These are utility functions which are re-used in many components.
"""

import numpy
from sklearn.metrics import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
	import mxnet as mx
except:
	print "ImportWarning: mxnet not imported"

from .blueberry import *

Q_LOWER_BOUND = 0.01
Q_UPPER_BOUND = 0.50
HIGH_FITHIC_CUTOFF = 10000000
LOW_FITHIC_CUTOFF =  25000
DATA_DIR = "/net/noble/vol1/home/jmschr/proj/contact/data"
DATA = lambda name: DATA_DIR + "/{}".format(name)

def negative_coordinate_pair( regions, contacts ):
	"""Returns a valid coordinate.

	Parameters
	----------
	regions : numpy.ndarray
		Valid mappable regions in the chromosome.

	contacts : dict
		Pairs of contacts

	Returns
	-------
	coordinates : tuple
		Region pair tuple
	"""

	while True:
		mid1, mid2 = numpy.random.choice(regions, 2)
		mid1, mid2 = min(mid1, mid2), max(mid1, mid2)

		if not (LOW_FITHIC_CUTOFF <= mid2 - mid1 <= HIGH_FITHIC_CUTOFF):
			continue

		if contacts.has_key( (mid1, mid2) ):
			continue

		break

	return mid1, mid2

def balanced_random_sample( regions, contacts ):
	"""Returns a balanced subset from the given contacts, regions, and restrictions.

	Parameters
	----------
	regions : numpy.ndarray
		Valid mappable regions in the chromosome.

	contacts : dict
		Pairs of contacts

	Returns
	-------
	coordinates : numpy.ndarray, shape=(n_contacts*2, 2)
		The coordinates for positive and negative samples

	y : numpy.ndarray, shape=(n_contacts*2,)
		Whether the coordinates indicate a contact or not.
	"""

	n = contacts.shape[0]
	coordinates = numpy.zeros((n*2, 2))
	y = numpy.concatenate((numpy.ones(n), numpy.zeros(n)))
	contact_dict = contacts_to_hashmap(contacts)

	coordinates[:n] = contacts
	for i in range(n):
		coordinates[i+n] = negative_coordinate_pair(regions, contact_dict)

	return coordinates, y

def benjamini_hochberg(p, alpha, n):
	"""Run the benjamini hochberg procedure on a vector of -sorted- p-values.

	Runs the procedure on a vector of p-values, and returns a mask of points
	which satisfy the q-value threshold specified by alpha.

	Parameters
	----------
	p : numpy.ndarray
		A vector of p values

	alpha : double, range=(0, 1)
		The final q-value we're going to return. Used in the step size.

	Returns
	-------
	mask : numpy.ndarray, shape=(p.shape[0],)
		boolean mask of values which satisfy the index.
	
	pval : double
		The threshold pvalue
	"""

	step = 1. * alpha / n
	mask = numpy.zeros(p.shape[0], dtype='int8')
	for i in xrange(p.shape[0]):
		if p[i] <= step*(i+1):
			mask[i] = 1

	return mask

def MxNetArray( path, inputs, shapes, dtypes, label=None, batch_size=1024 ):
	"""Returns a properly formatted mxnet data iterator. 

	Parameters
	----------
	path : str
		Location of the data

	inputs : array-like of strings 
		The inputs to use. Typically 'x1seq' 'x1dnase' 'x2seq' 'x2dnase'

	shapes : array-like of tuples
		The shape of the underlying data points

	dtypes : array-like of strings
		The type of the underlying data

	label : str, optional
		The name of the label file. If none, does not add a label.

	batch_size : int
		The number of points per batch to use.

	Returns
	-------
	data : mxnet.io.NDArrayIter
		The data in a properly formatted way.
	"""

	X = { name : mmap( path + name + '.npy', shape=shape, dtype=dtype ) 
			for name, shape, dtype in zip(inputs, shapes, dtypes) }

	if label is not None:
		y = mmap( path + label + '.npy', (), dtype='float32' )
		data = mx.io.NDArrayIter( X, label={'softmax_label' : y}, batch_size=batch_size )
	else:
		data = mx.io.NDArrayIter( X, batch_size=batch_size )

	return data

def plot_pr_auc(y_true, y_pred, outfile="pr_auc.png"):
	"""Plot a ROC and PR curve on the same plot.

	Parameters
	----------
	y_true : array-like, int
		The true binary labels of the data

	y_pred : array-like, double
		The predicted values of the data used for ranking.

	Returns
	-------
	None
	"""

	plt.figure( figsize=(15, 6) )
	fpr, tpr, _ = roc_curve( y_true, y_pred )
	precision, recall, _ = precision_recall_curve( y_true, y_pred )
	eval_auc = numpy.around( roc_auc_score( y_true, y_pred ), 4 )

	print "ROC AUC: {}".format(eval_auc)

	plt.subplot(121)
	plt.title( "ROC", fontsize=16 )
	plt.xlabel( "FPR", fontsize=14 )
	plt.ylabel( "TPR", fontsize=14 )
	plt.plot( fpr, tpr, c='c', label="AUC = {}".format( eval_auc ) )
	plt.plot( [0,1], [0, 1], c='k', alpha=0.6 )
	plt.legend( loc=4 )

	eval_auc = numpy.around( average_precision_score( y_true, y_pred ), 4 )

	print "PR AUC: {}".format(eval_auc)

	plt.subplot(122)
	plt.title( "Precision-Recall", fontsize=16 )
	plt.xlabel( "Recall", fontsize=14 )
	plt.ylabel( "Precision", fontsize=14 )
	plt.plot( recall, precision, c='c', label="AUC = {}".format( eval_auc ) )
	plt.plot( [0,1], [0.5, 0.5], c='k', alpha=0.6 )
	plt.ylim(0.0, 1.0)
	plt.legend( loc=4 )
	plt.savefig(outfile)
