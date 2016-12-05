# blueberry.pyx
# Contact: Jacob Schreiber (jmschreiber91@gmail.com)

"""
Cython-optimized functions for converting data into a LMDB database
in a flexible manner.
"""

from libc.math cimport exp
cimport numpy
import numpy
import random, time, os, gzip
from .utils import *

from joblib import Parallel, delayed

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

def predict(name, iteration, celltype='GM12878', use_seq=True, 
	use_dnase=True, use_dist=True, min_dist=50000, max_dist=1000000, 
	batch_size=10240):

	ctxs = [0, 1, 2, 3]

	Parallel(n_jobs=4)( delayed(predict_task)(name, iteration, ctx, 4, celltype,
		use_seq, use_dnase, use_dist, min_dist, max_dist, batch_size) for ctx in ctxs)

	resolution = 1000
	width = 500
	n = numpy.load('chr21.pred.npy', mmap_mode='r').shape[0]
	y = numpy.zeros((n, n))

	for ctx in ctxs:
		with open('{}-{}-{}-{}-predictions.txt'.format(name, iteration, celltype, ctx), 'r') as infile:
			for line in infile:
				mid1, mid2, p = line.split()
				mid1 = (int(float(mid1)) - width) / resolution
				mid2 = (int(float(mid2)) - width) / resolution
				p = float(p)

				y[mid1, mid2] = p

	numpy.save("chr21.{}.y_pred.1000.npy".format(celltype), y)
	os.system('rm {}-{}-{}-*-predictions.txt'.format(name, iteration, celltype))	

def predict_task(name, iteration, ctx, n_jobs, celltype='GM12878', bint use_seq=True, bint use_dnase=True, 
	bint use_dist=True, int min_dist=50000, int max_dist=1000000, batch_size=10240):
	cdef int window = 1000, width = 500
	cdef int k = 0, tot = 0, i, j, l, mid1, mid2
	cdef numpy.ndarray sequence = numpy.load('/data/scratch/ssd/jmschr/contact/chr21.ohe.npy')
	cdef numpy.ndarray dnase = numpy.load('/data/scratch/ssd/jmschr/contact/chr21.{}.ohe_dnase.npy'.format(celltype))
	cdef numpy.ndarray regions = numpy.load('/data/scratch/ssd/jmschr/contact/chr21.GM12878.regions.1000.npy').astype('int')
	cdef numpy.ndarray predictions = numpy.zeros((batch_size, 3), dtype='float32')
	cdef int n = regions.shape[0]

	print "GPU [{}] [{}] -- data loaded".format(celltype, ctx)
	model = mx.model.FeedForward.load(name, iteration, ctx=mx.gpu(ctx))
	print "GPU [{}] [{}] -- model loaded".format(celltype, ctx)

	with open('{}-{}-{}-{}-predictions.txt'.format(name, iteration, celltype, ctx), 'w') as outfile:
		for mid1 in regions:
			for mid2 in regions[ctx::n_jobs]:
				if not min_dist <= mid2 - mid1 <= max_dist:
					continue

				if k == 0:
					data = { 'x1seq'    : numpy.zeros((batch_size, window, 4)),
							 'x2seq'    : numpy.zeros((batch_size, window, 4)),
							 'x1dnase'  : numpy.zeros((batch_size, window, 8)),
							 'x2dnase'  : numpy.zeros((batch_size, window, 8)),
							 'distance' : numpy.zeros((batch_size, 191)) }

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

					predictions[k, 0] = mid1
					predictions[k, 1] = mid2

					k += 1
					tot += 1

				else:
					print "GPU [{}] [{}] -- {} samples loaded, predicting...".format(celltype, ctx, k),
					data['x1seq'] = data['x1seq'].reshape((batch_size, 1, window, 4))
					data['x2seq'] = data['x2seq'].reshape((batch_size, 1, window, 4))
					data['x1dnase'] = data['x1dnase'].reshape((batch_size, 1, window, 8))
					data['x2dnase'] = data['x2dnase'].reshape((batch_size, 1, window, 8))

					X = mx.io.NDArrayIter(data, batch_size=1024)
					y = model.predict(X)
					k = 0

					data['x1seq'] = data['x1seq'].reshape((batch_size, window, 4))
					data['x2seq'] = data['x2seq'].reshape((batch_size, window, 4))
					data['x1dnase'] = data['x1dnase'].reshape((batch_size, window, 8))
					data['x2dnase'] = data['x2dnase'].reshape((batch_size, window, 8))

					predictions[:,2] = y[:,1]
					for mid1, mid2, y in predictions:
						outfile.write( "{} {} {}\n".format(mid1, mid2, y) )

					predictions *= 0

					print
					print "GPU [{}] [{}] -- {} samples predicted and output".format(celltype, ctx, tot)
