# blueberry.pyx
# Contact: Jacob Schreiber (jmschreiber91@gmail.com)

from libc.math cimport exp
cimport numpy
import numpy
import random, time, os, gzip
from .utils import *

from joblib import Parallel, delayed

random.seed(0)
numpy.random.seed(0)

cpdef insulation_score(numpy.ndarray y_pred, int size=100):
	cdef int i, j, k, n = y_pred.shape[0]
	cdef numpy.ndarray y = numpy.zeros(n)
	cdef numpy.ndarray sums = y_pred.sum(axis=0)

	for i in range(n):
		if sums[i] > 0:
			for j in range(-size, size+1):
				if i+j >= n:
					break

				for k in range(j, size+1):
					if i+k >= n:
						break

					y[i] += y_pred[i+j, i+k]

	return y

def benjamini_hochberg(numpy.ndarray p_values, long n):
	"""Run the benjamini hochberg procedure on a vector of -sorted- p-values.

	Runs the procedure on a vector of p-values, and returns the q-values for
	each point.

	Parameters
	----------
	p_values : numpy.ndarray
		A vector of p values

	n : int
		The number of tests which have been run.

	Returns
	-------
	q_values : numpy.ndarray
		The q-values for each point.
	"""

	p_values = p_values.astype('float64')

	cdef long i, d = p_values.shape[0]
	cdef double q_value, prev_q_value = 0.0

	cdef numpy.ndarray q_values = numpy.zeros_like(p_values)

	for i in range(d):
		q_value = p_values[i] * n / (i+1)
		q_value = min(q_value, 1)
		q_value = max(q_value, prev_q_value)

		q_values[i] = q_value
		prev_q_value = q_value

	return q_values

cpdef count_band_regions( numpy.ndarray regions_ndarray ):
	"""Calculate the number of regions in the band."""

	cdef double* regions = <double*> regions_ndarray.data
	cdef int n = regions_ndarray.shape[0], i, j
	cdef int HIGH = HIGH_FITHIC_CUTOFF, LOW = LOW_FITHIC_CUTOFF
	cdef long t = 0

	with nogil:
		for i in range(n):
			for j in range(i):
				if LOW <= regions[i] - regions[j] <= HIGH:
					t += 1

	return t

cpdef numpy.ndarray downsample(numpy.ndarray x, numpy.ndarray regions,
	int min_dist=50000, int max_dist=1000000):
	cdef int mid1, mid2, i, j, k1, k2, l1, l2
	cdef numpy.ndarray y = numpy.zeros((9626, 9626))

	for mid1 in regions:
		for mid2 in regions:
			if min_dist <= mid2 - mid1 <= max_dist:
				k1 = (mid1 - 500) / 1000
				k2 = (mid2 - 500) / 1000
				l1 = (mid1 - 2500) / 5000
				l2 = (mid2 - 2500) / 5000

				for i in range(-2, 3):
					for j in range(-2, 3):
						y[l1, l2] = max(y[l1, l2], x[k1 + i, k2 + j])

	return y
