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
from .datatypes import *

Q_LOWER_BOUND = 0.01
Q_UPPER_BOUND = 0.50
HIGH_FITHIC_CUTOFF = 10000000
LOW_FITHIC_CUTOFF = 25000
DATA_DIR = "/net/noble/vol1/home/jmschr/proj/contact/data"
DATA = lambda name: DATA_DIR + "/{}".format(name)


def extract_contacts(celltype, chromosome, resolution, alpha=None, n_regions=None, filename=None):
	"""Extract contacts from a given chromosome, and number of regions in the band.

	Extract all regions which have a p-value <= alpha. This is useful for the
	Benjamini-Hochkin q-value assignment. Also return the number of possible positions.

	Parameters
	----------
	celltype : str
		The celltype to use.

	chromosome : int
		The chromosome to analyze.

	resolution : int
		The resolution to use. Usually either 1000 or 5000.

	alpha : double, range=(0, 1)
		The final q-value we're going to return. Can filter out all results higher
		than this.

	Returns
	-------
	contacts : numpy.ndarray, shape=(n_contacts, 4)
		The contacts. Columns are the chromosome, mid1, mid2, and p-value.

	n : int
		Number of possible positions in the band.
	"""

	print "CPU [{}]: Extracting {} chr{}".format(chromosome, celltype, chromosome)

	try:
		contactMap = FithicContactMap(celltype, chromosome, resolution, filename)
		contact = contactMap.map
	except Exception as e:
		print "CPU [{}]: {}".format(chromosome, e.message)
		return numpy.zeros((0, 5)), 0

	# Columns normally correspond to mid1, mid2, contactCount, p, q
	# Remove points not satisfying alpha to make it smaller ASAP
	if alpha is not None:
		contact = contact[contact[:,3] <= alpha]

	# Shift everything over by a column
	contact[:,1:] = contact[:,:-1]
	contact[:,0] = chromosome

	# Calculate distances
	distances = contact[:,2] - contact[:,1]

	# Filter out points which aren't in the band
	contact = contact[(distances <= HIGH_FITHIC_CUTOFF) & (distances >= LOW_FITHIC_CUTOFF)]

	# Remove genomic distance. These points are now ordered as
	# chromosome, mid1, mid2, contactCount, p
	if n_regions:
		return contact, count_band_regions(contactMap.regions)
	else:
		return contact

def negative_coordinate_pair(regions, contacts, min_dist=25000, max_dist=10000000):
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

		if not (min_dist <= mid2 - mid1 <= max_dist):
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

def plot_roc_pr(y_true, y_pred, labels, styles=None):
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

	colors = styles or 'cmrbgky'
	plt.figure( figsize=(16, 6) )

	for c, yp, label in zip(colors, y_pred, labels):
		fpr, tpr, _ = roc_curve(y_true, yp)
		eval_auc = numpy.around(roc_auc_score(y_true, yp), 4)

		plt.subplot(121)
		plt.title("ROC", fontsize=16)
		plt.xlabel("FPR", fontsize=14)
		plt.ylabel("TPR", fontsize=14)
		plt.plot(fpr, tpr, c, label="{}: {}".format(label, eval_auc))

		precision, recall, _ = precision_recall_curve(y_true, yp)
		eval_auc = numpy.around(average_precision_score(y_true, yp), 4)

		plt.subplot(122)
		plt.title("Precision-Recall", fontsize=16)
		plt.xlabel("Recall", fontsize=14)
		plt.ylabel("Precision", fontsize=14)
		plt.plot(recall, precision, c, label="{}: {}".format(label, eval_auc))
		plt.ylim(0.0, 1.0)

	plt.subplot(121)
	plt.plot([0,1], [0, 1], c='k', alpha=0.6)
	plt.legend(loc=4, fontsize=14)

	y = y_true.mean()
	y = min(y, 1-y)
	plt.subplot(122)
	plt.plot([0,1], [y, y], c='k', alpha=0.6)
	plt.legend(loc=4, fontsize=14)
