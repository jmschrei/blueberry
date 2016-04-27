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

cpdef evaluate( map1, map2 ):
	"""Calculate the accuracy using map1 to predict map2."""

	regions = numpy.array(list(set(map1.regions).intersection(set(map2.regions))))
	regions.sort()
	
	cdef int n = regions.shape[0]**2/2, m = regions.shape[0]
	cdef numpy.ndarray y_true = numpy.zeros(n)
	cdef numpy.ndarray y_pred = numpy.zeros(n)

	cdef int i=0, j=0, k, l

	for k in range(m):
		mid1 = regions[k]

		for l in range(m):
			if l <= k:
				continue

			mid2 = regions[l]

			j += 1
			if j % 1000000 == 0:
				print i, j, n

			if mid2 > mid1 + HIGH_FITHIC_CUTOFF:
				break
			elif mid2 < mid1 + LOW_FITHIC_CUTOFF:
				continue
			
			y_pred[i] = map1.get((mid1, mid2), (1, 1))[0]
			y_true[i] = 0 if map2.get((mid1, mid2), (1, 1))[1] > 0.01 else 1
			i += 1

	print i, j, n
	y_true = y_true[:i]
	y_pred = -numpy.log(y_pred[:i])
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

cpdef tuple extract_regions( int [:] x, numpy.ndarray chromosome, numpy.ndarray dnase, int window ):
	"""Extract regions."""

	cdef numpy.ndarray seq   = numpy.array(extract_sequences( chromosome, x, window ))
	cdef numpy.ndarray dnases = numpy.array(extract_dnases( dnase, x, window ))	
	return seq, dnases

cdef void memmap_extract_sequences( float [:,:] chromosome, int [:] centers, int window, str name ):
	"""Extract the sequences and save them to a memory map."""

	print "-> Saving '{}'".format( name )
	cdef int i, n = centers.shape[0]
	cdef int breadth = 2*window+1
	cdef int center

	seqs = numpy.memmap( name, dtype='float32', mode='w+', shape=(n, breadth, 4))

	for i in range(n):
		center = edge_correct( centers[i], chromosome.shape[0], chromosome.shape[0], window )
		seqs[i] = chromosome[center-window:center+window+1]

		if i % 50000 == 0:
			print "\tFlushing '{}' with {} items".format(name, i)
			seqs.flush()

	seqs.flush()
	print "<- Done Saving '{}'".format( name )


cdef void memmap_extract_dnases( float [:] dnase, int [:] centers, int window, str name ):
	"""Extract the dnase and save them to a memory map."""

	print "-> Saving '{}'".format( name )
	cdef int i, n = centers.shape[0]
	cdef int breadth = 2*window+1
	cdef int center

	dnases = numpy.memmap( name, dtype='float32', mode="w+", shape=(n, breadth, 1))

	for i in range(n):
		center = edge_correct( centers[i], dnase.shape[0], dnase.shape[0], window )
		dnases[i,:,0] = dnase[center-window:center+window+1]

		if i % 50000 == 0:
			print "\tFlushing '{}' with {} items".format(name, i)
			dnases.flush()

	dnases.flush()
	print "<- Done Saving '{}'".format( name )

cpdef extract_full_dataset( chrom, window=500 ):
	"""Extract the dataset given a chromosome."""

	DATA_DIR = '/data/scratch/ssd/jmschr/contact/'

	chromosome = numpy.load( '../data/chr{}.ohe.npy'.format(chrom) ).astype( 'float32' )
	dnase      = numpy.load( '../data/chr{}.dnase.npy'.format(chrom) ).astype( 'float32' )
	positive   = numpy.load( '../data/chr{}.full.positive_contacts.npy'.format(chrom), mmap_mode='r' )
	negative   = numpy.load( '../data/chr{}.full.negative_contacts.npy'.format(chrom), mmap_mode='r' )

	contacts = numpy.concatenate((positive, negative))
	n = contacts.shape[0]
	indices = numpy.arange(n)
	numpy.random.shuffle(indices)

	contacts = contacts[indices]

	numpy.save( DATA_DIR + 'chr{}.full.x1coord.npy'.format(chrom), contacts[:,0] )
	numpy.save( DATA_DIR + 'chr{}.full.x2coord.npy'.format(chrom), contacts[:,1] )

	memmap_extract_sequences( chromosome, contacts[:,0], window, DATA_DIR + 'chr{}.full.x1seq.npy'.format(chrom) )
	memmap_extract_sequences( chromosome, contacts[:,1], window, DATA_DIR + 'chr{}.full.x2seq.npy'.format(chrom) )

	memmap_extract_dnases( dnase, contacts[:,0], window, DATA_DIR + 'chr{}.full.x1dnase.npy'.format(chrom) )
	memmap_extract_dnases( dnase, contacts[:,1], window, DATA_DIR + 'chr{}.full.x2dnase.npy'.format(chrom) )

	y = numpy.concatenate(( numpy.ones(positive.shape[0], dtype='float32'), numpy.zeros(negative.shape[0], dtype='float32') ))
	y = y[indices]

	numpy.save( DATA_DIR + 'chr{}.full.y.npy'.format(chrom), y )

	os.system( 'mv {}chr{}* /net/noble/vol1/home/jmschr/proj/contact/data/'.format(DATA_DIR, chrom) )
