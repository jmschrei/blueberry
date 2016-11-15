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

cpdef downsample(float[:, :] yp1, float[:, :] yp5, float[:, :] yp5i):
	cdef int i, j, k, ni, nj
	cdef int n5 = yp5.shape[0]

	for i in range(n5-1):
		for j in range(n5-1):

			for ni in range(i*5, (i+1)*5):
				for nj in range(j*5, (j+1)*5):
					yp5i[i, j] = max(yp5i[i, j], yp1[ni, nj])

	return numpy.array(yp5i)

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

cpdef dict region_dict(numpy.ndarray regions, min_dist, max_dist):
	cdef int i, j, n = regions.shape[0]
	cdef double x, y
	cdef dict region_dict = {}

	for i in range(n):
		x = regions[i]
		region_dict[x] = []

		for j in range(i, n):
			y = regions[j]

			if min_dist <= y - x <= max_dist:
				region_dict[x].append(y)

	return region_dict


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
