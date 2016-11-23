# blueberry.pyx
# Contact: Jacob Schreiber (jmschreiber91@gmail.com)

"""
Cython-optimized functions for converting data into a LMDB database
in a flexible manner.
"""

from libc.math cimport exp

cimport numpy
import numpy
from scipy.sparse import csr_matrix, lil_matrix
from scipy import io


import random, time, os, gzip
from .utils import *

random.seed(0)
numpy.random.seed(0)

cpdef numpy.ndarray translate( numpy.ndarray sequence, dict mapping ):
	"""Translate the sequence from 
	Translate the sequence. Since this involves a lot of lookups, it's faster
	to do it in cython.
	"""

	cdef int i, idx
	cdef int n = sequence.shape[0]

	ohe_sequence = numpy.zeros( (n, 4), dtype=numpy.int32 )

	for i in range(n):
		if sequence[i] == 'N':
			continue

		idx = mapping[ sequence[i] ]
		ohe_sequence[i, idx] = 1

	return ohe_sequence

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

def predict(model, n_bins, outfile, bint use_seq=True, bint use_dnase=True, 
	bint use_dist=True, int min_dist=25000, int max_dist=10000000):
	cdef int batch_size = 10240, window = 1000, width = 500
	cdef int k = 0, tot = 0, i, j, l, mid1, mid2, coord1, coord2
	cdef numpy.ndarray sequence = numpy.load('/data/scratch/ssd/jmschr/contact/chr21.ohe.npy')
	cdef numpy.ndarray dnase = numpy.load('/data/scratch/ssd/jmschr/contact/chr21.GM12878.ohe_dnase.npy')
	cdef numpy.ndarray regions = numpy.load('/data/scratch/ssd/jmschr/contact/chr21.GM12878.regions.1000.npy').astype('int')

	cdef numpy.ndarray coords = numpy.zeros((batch_size, 2))
	cdef numpy.ndarray predictions = numpy.zeros((n_bins, n_bins), dtype='float32')

	cdef int n = regions.shape[0]

	#model = mx.model.Module.load(name, iteration, ctx=[mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)])

	for i in range(n):
		mid1 = regions[i]
		for j in range(i, n):
			mid2 = regions[j]
			if not min_dist <= mid2 - mid1 <= max_dist:
				continue

			if k == 0:
				data = { 'x1seq'    : numpy.zeros((batch_size, window, 4)),
						 'x2seq'    : numpy.zeros((batch_size, window, 4)),
						 'x1dnase'  : numpy.zeros((batch_size, window, 8)),
						 'x2dnase'  : numpy.zeros((batch_size, window, 8)),
						 'distance' : numpy.zeros((batch_size, 281)) }

			if k != batch_size:
				if use_seq:
					data['x1seq'][k] = sequence[mid1-width:mid1+width]
					data['x2seq'][k] = sequence[mid2-width:mid2+width]

				if use_dnase:
					data['x1dnase'][k] = dnase[mid1-width:mid1+width]
					data['x2dnase'][k] = dnase[mid2-width:mid2+width]

				if use_dist:
					distance = mid2 - mid1 - min_dist
					for l in range(100):
						data['distance'][k][l] = 1 if distance >= l*1000 else 0
					for l in range(91):
						data['distance'][k][l+100] = 1 if distance >= 100000 + l*10000 else 0
					for l in range(91):
						data['distance'][k][l+190] = 1 if distance >= 1000000 + l*100000 else 0

				coords[k] = mid1, mid2

				k += 1
				tot += 1

			else:
				print "[GPU] -- {} samples loaded, predicting...".format(k),
				data['x1seq'] = data['x1seq'].reshape((batch_size, 1, window, 4))
				data['x2seq'] = data['x2seq'].reshape((batch_size, 1, window, 4))
				data['x1dnase'] = data['x1dnase'].reshape((batch_size, 1, window, 8))
				data['x2dnase'] = data['x2dnase'].reshape((batch_size, 1, window, 8))

				X = mx.io.NDArrayIter(data, batch_size=1024)
				y = model.predict(X)

				data['x1seq'] = data['x1seq'].reshape((batch_size, window, 4))
				data['x2seq'] = data['x2seq'].reshape((batch_size, window, 4))
				data['x1dnase'] = data['x1dnase'].reshape((batch_size, window, 8))
				data['x2dnase'] = data['x2dnase'].reshape((batch_size, window, 8))

				for l in range(k):
					coord1, coord2 = coords[l]
					distance = coord2 - coord1

					coord1 = (coord1 - width) / window
					coord2 = (coord2 - width) / window

					if 25000 <= distance <= 100000:
						predictions[coord1, coord2] = y[0][l, 1]
					elif 100000 < distance <= 1000000:
						predictions[coord1, coord2] = y[1][l, 1]
					else:
						predictions[coord1, coord2] = y[2][l, 1]

				k = 0
				coords *= 0

				print
				print "[GPU] -- {} samples predicted and output".format(tot)

	numpy.save(outfile, predictions)
