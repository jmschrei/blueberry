# utils.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

"""
These are utility functions which are re-used in many components.
"""

from sklearn.metrics import *

Q_LOWER_BOUND = 0.01
Q_UPPER_BOUND = 0.50
HIGH_FITHIC_CUTOFF = 10000000
LOW_FITHIC_CUTOFF = 10000

def mmap( name, shape, dtype='int8' ):
	"""Return a properly formatted memory map.

	This will read in a memory map as a 1d array of numbers, and return it
	properly formatted.

	Parameters
	----------
	name : str
		The path to the memory map to open.

	shape : tuple
		The shape of each element in the memory map. For example, data which is
		eventually (5, 1, 4, 1) would be have a shape of (1, 4, 1).

	dtype : str, optional
		The data type to read in. Must be a numpy dtype.

	Returns
	------
	mmap : numpy.memmap
		The numpy memory map of properly formatted data.
	"""

	data = numpy.memmap( name, dtype=dtype, mode='r' )
	k = numpy.prod(shape)
	n = int(data.shape[0] / k)
	return data.reshape(n, *shape)

def plot_pr_auc(y_true, y_pred, outfile="roc_pr.png"):
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
	plt.ylim(0.5, 1.0)
	plt.legend( loc=4 )
