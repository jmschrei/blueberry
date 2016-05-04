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

cpdef evaluate( map1, map2, low_cutoff=None, high_cutoff=None, return_dist=False ):
	"""Calculate the accuracy using map1 to predict map2."""

	low_cutoff = low_cutoff or LOW_FITHIC_CUTOFF
	high_cutoff = high_cutoff or HIGH_FITHIC_CUTOFF 

	regions = numpy.array(list(set(map1.regions).intersection(set(map2.regions))))
	regions.sort()
	
	print map1.regions.shape
	print map2.regions.shape

	d1 = contacts_to_qhashmap( map1.map )
	d2 = contacts_to_qhashmap( map2.map )

	cdef int n = regions.shape[0]**2/2, m = regions.shape[0]
	cdef numpy.ndarray y_true = numpy.zeros(n)
	cdef numpy.ndarray y_pred = numpy.zeros(n)
	cdef numpy.ndarray dist = numpy.zeros(n)

	cdef int i=0, j=0, k, l

	for k in range(m):
		mid1 = regions[k]

		for l in range(k, m):
			mid2 = regions[l]

			j += 1
			if j % 1000000 == 0:
				print i, j, n

			if mid2 > mid1 + high_cutoff:
				break
			elif mid2 < mid1 + low_cutoff:
				continue
			
			y_pred[i] = d1.get((mid1, mid2), (1, 1))[0]
			y_true[i] = 0 if d2.get((mid1, mid2), (1, 1))[1] > 0.01 else 1
			dist[i] = mid2 - mid1
			i += 1

	print i, j, n
	y_true = y_true[:i]
	y_pred = -numpy.log(y_pred[:i])

	if return_dist:
		return y_true, y_pred, dist[:i]
	return y_true, y_pred

cpdef dict contacts_to_hashmap( numpy.ndarray contacts ):
	"""Take in pairs of contacts and build a hashmap of tuples for fast lookup."""

	cdef int i
	cdef int n = contacts.shape[0]
	cdef int mid1, mid2
	cdef dict contact_map = {}

	for i in range(n):
		mid1, mid2 = contacts[i]
		mid1 = int(mid1)
		mid2 = int(mid2)

		contact_map[(mid1, mid2)] = 1
		contact_map[(mid2, mid1)] = 1

	return contact_map

cpdef dict contacts_to_qhashmap( numpy.ndarray contacts ):
	"""Take in pairs of contacts and build a hashmap of tuples for fast lookup."""

	cdef int i
	cdef int n = contacts.shape[0]
	cdef int mid1, mid2
	cdef double p, q

	cdef dict contact_map = {}

	for i in range(n):
		mid1, mid2, p, q = contacts[i]
		mid1 = int(mid1)
		mid2 = int(mid2)

		contact_map[(mid1, mid2)] = (p, q)
		contact_map[(mid2, mid1)] = (p, q)

	return contact_map

cdef inline int edge_correct( int peak, int n, int m, int l ):
	if peak >= n-l-1:
		peak = n-l-1
	if peak >= m-l-1:
		peak = m-l-1
	return peak

cdef numpy.ndarray extract_sequences( int [:,:] chromosome, int [:] centers, int window ):
	"""Extract a window of size 251 from the chromosome, centered around the peak."""

	cdef int i, n = centers.shape[0]
	cdef numpy.ndarray seqs = numpy.zeros((n, 2*window+1, 4), dtype=numpy.int32)

	for i in range(n):

		seqs[i] = chromosome[centers[i]-window:centers[i]+window+1]

	return seqs

cdef numpy.ndarray extract_dnases( float [:] dnase, int [:] centers, int window ):
	"""Extract a window of size 251 from the chromosome, centered around the peak."""

	cdef int i, n = centers.shape[0]
	cdef numpy.ndarray dnases = numpy.zeros((n, 2*window+1, 1), dtype=numpy.float64)

	for i in range(n):
		dnases[i,:,0] = dnase[centers[i]-window:centers[i]+window+1]

	return dnases

cpdef tuple extract_regions( numpy.ndarray x, numpy.ndarray chromosome, numpy.ndarray dnase, int window ):
	"""Extract regions."""

	cdef numpy.ndarray seq   = numpy.array(extract_sequences( chromosome, x, window ))
	cdef numpy.ndarray dnases = numpy.array(extract_dnases( dnase, x, window ))	
	return seq, dnases

