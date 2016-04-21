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

cpdef dict contacts_to_hashmap( numpy.ndarray contacts ):
	"""Take in pairs of contacts and build a hashmap of tuples for fast lookup."""

	cdef int i
	cdef int n = contacts.shape[0]
	cdef int mid1, mid2

	cdef dict contact_map = {}

	for i in range(n):
		mid1 = contacts[i, 0]
		mid2 = contacts[i, 1]

		contact_map[ (mid1, mid2) ] = 1
		contact_map[ (mid2, mid1) ] = 1

	return contact_map

cpdef dict contacts_to_qhashmap( numpy.ndarray contacts ):
	"""Take in pairs of contacts and build a hashmap of tuples for fast lookup."""

	cdef int i
	cdef int n = contacts.shape[0]
	cdef int mid1, mid2
	cdef double q

	cdef dict contact_map = {}

	for i in range(n):
		mid1 = int(contacts[i, 0])
		mid2 = int(contacts[i, 1])
		q = contacts[i, 2]

		contact_map[ (mid1, mid2) ] = q
		contact_map[ (mid2, mid1) ] = q

	return contact_map

cpdef numpy.ndarray contact_map( numpy.ndarray contacts, dict positive_contacts, dict negative_contacts ):
	"""
	Build an example matrix of intrachromosomal contacts for the entire genome
	and return examples one at a time as a generator.
	"""

	cdef int i, j = 0
	cdef int n = contacts.shape[0]
	cdef numpy.ndarray contact_map = numpy.zeros((n, 3), dtype=numpy.int32)

	cdef int mid1, mid2, mid1_round, mid2_round
	cdef int contact_count = 0, noncontact_count = 0, filtered_count = 0
	cdef int contact
	cdef tuple key

	for i in range(n):
		if i % 25000 == 0:
			print "\t\t{} contacts, {} non-contacts, {} filtered".format( contact_count, noncontact_count, filtered_count )

		mid1 = contacts[i, 0]
		mid2 = contacts[i, 1]

		mid1_round = numpy.around( mid1+500, -3 ) - 500
		mid2_round = numpy.around( mid2+500, -3 ) - 500

		key = (mid1_round, mid2_round)

		if positive_contacts.has_key( key ):
			contact = 1
			contact_count += 1
		elif negative_contacts.has_key( key ):
			contact = 0
			noncontact_count += 1
		else:
			filtered_count += 1
			continue

		contact_map[j, 0] = mid1
		contact_map[j, 1] = mid2
		contact_map[j, 2] = contact
		j += 1

	return contact_map[:j]


cpdef numpy.ndarray build_all_contact_map( numpy.ndarray contacts, dict positive_contacts ):
	"""
	Build an example matrix of intrachromosomal contacts for the entire genome
	and return examples one at a time as a generator.
	"""

	cdef int i, j, k
	cdef int n = contacts.shape[0]
	cdef numpy.ndarray contact_map = numpy.zeros( (n, 3), dtype=numpy.int32 )

	cdef int mid1, mid2, mid1_round, mid2_round
	cdef int contact_count = 0
	cdef int noncontact_count = 0
	cdef int contact

	for i in range(n):
		if i % 25000 == 0:
			print "\t\t{} contacts, {} non-contacts".format( contact_count, noncontact_count )

		mid1 = contacts[i, 0]
		mid2 = contacts[i, 1]

		mid1_round = numpy.around( mid1+500, -3 ) - 500
		mid2_round = numpy.around( mid2+500, -3 ) - 500

		contact = positive_contacts.has_key( (mid1_round, mid2_round) )

		if contact == 1:
			contact_count += 1
		else:
			noncontact_count += 1

		contact_map[i, 0] = mid1
		contact_map[i, 1] = mid2
		contact_map[i, 2] = contact

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

cpdef numpy.ndarray extract_pair_subset( numpy.ndarray contacts, numpy.ndarray peaks ):
	"""
	Given a numpy array of all contacts, and a numpy array of relevant peaks
	which we want the contacts between, extract all contacts with those peaks.
	"""

	cdef int i, n = peaks.shape[0], j = 0
	cdef dict peak_indices = {}
	cdef numpy.ndarray contact_subset = numpy.zeros((n*n, 3))

	for i in range(n):
		peak_indices[ peaks[i] ] = i

	for i in range( contacts.shape[0] ):
		peak1 = contacts[i, 0]
		peak2 = contacts[i, 1]
		contact = contacts[i, 2]

		if peak_indices.has_key( peak1 ) and peak_indices.has_key( peak2 ):
			contact_subset[j, 0] = peak1
			contact_subset[j, 1] = peak2
			contact_subset[j, 2] = contact
			j += 1

	return contact_subset[:j]

cpdef tuple extract_regions( int [:] x, numpy.ndarray chromosome, numpy.ndarray dnase, int window ):
	"""Extract regions."""

	cdef numpy.ndarray seq   = numpy.array(extract_sequences( chromosome, x, window ))
	cdef numpy.ndarray dnases = numpy.array(extract_dnases( dnase, x, window ))	
	return seq, dnases

cpdef tuple extract_pairs( numpy.ndarray regions, numpy.ndarray contacts, numpy.ndarray chromosome, numpy.ndarray dnase, int window ):
	"""Encode pairwise contacts using pre-extracted sites."""

	cdef int i, j, k=0, n=regions.shape[0]
	cdef int m = n*n
	cdef int label

	cdef numpy.ndarray x1seq   = numpy.zeros((m, 1, 2*window+1, 4))
	cdef numpy.ndarray x1dnase = numpy.zeros((m, 1, 2*window+1, 1))
	cdef numpy.ndarray x2seq   = numpy.zeros((m, 1, 2*window+1, 4))
	cdef numpy.ndarray x2dnase = numpy.zeros((m, 1, 2*window+1, 1))
	cdef numpy.ndarray y       = numpy.zeros((m,))

	for i in range(n):
		regions[i] = edge_correct(regions[i], chromosome.shape[0], dnase.shape[0], window)

	regions = regions.astype(numpy.int32)
	chromosome = chromosome.astype(numpy.int32)
	dnase = dnase.astype(numpy.float32)

	cdef numpy.ndarray seqs   = extract_sequences( chromosome, regions, window )
	cdef numpy.ndarray dnases = extract_dnases( dnase, regions, window )

	cdef dict positive = contacts_to_hashmap(contacts)
	cdef tuple contact

	k = 0
	for i in range(n):
		for j in range(i):
			contact = (regions[i], regions[j])
			label = positive.get( contact, 0 )
			k += label
			print regions[i], regions[j], label, k

			continue

			x1seq[k]   = seqs[j]
			x2seq[k]   = seqs[i]
			x1dnase[k] = dnases[j]
			x2dnase[k] = dnases[i]
			y[k]       = label

	return x1seq, x1dnase, x2seq, x2dnase, y

def extract_dataset( chrom_id, window=500 ):
	"""Extract the dataset from a given chromosome."""

	chromosome = numpy.load( '../data/chr{}.ohe.npy'.format( chrom_id ) )
	contacts = numpy.load( '../data/chr{}.peak_contacts.npy'.format( chrom_id ) )
	peaks = numpy.load( '../data/chr{}.peaks.npy'.format( chrom_id ) )
	dnase = numpy.load( '../data/chr{}.dnase.npy'.format( chrom_id ) )

	positive_contacts = contacts[ contacts[:,2] == 1 ]
	negative_contacts = contacts[ contacts[:,2] == 0 ]
	numpy.random.shuffle( negative_contacts )

	n = positive_contacts.shape[0]

	contacts = numpy.concatenate( (positive_contacts, negative_contacts[:n]) )
	numpy.random.shuffle( contacts )
	
	x1seq, x1dnase, x2seq, x2dnase, y = extract_pairs( contacts, peaks, chromosome, dnase, window )

	x1seq = x1seq.astype('float32')
	x2seq = x2seq.astype('float32')
	x1dnase = x1dnase.astype('float32')
	x2dnase = x2dnase.astype('float32')
	y = y.astype('float32')
	return x1seq, x2seq, x1dnase, x2dnase, y

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

	numpy.save( DATA_DIR + 'chr{}.full.x1coord.npy'.format(chrom), contacts[:,0] )
	numpy.save( DATA_DIR + 'chr{}.full.x2coord.npy'.format(chrom), contacts[:,1] )

	memmap_extract_sequences( chromosome, contacts[:,0], window, DATA_DIR + 'chr{}.full.x1seq.npy'.format(chrom) )
	memmap_extract_sequences( chromosome, contacts[:,1], window, DATA_DIR + 'chr{}.full.x2seq.npy'.format(chrom) )

	memmap_extract_dnases( dnase, contacts[:,0], window, DATA_DIR + 'chr{}.full.x1dnase.npy'.format(chrom) )
	memmap_extract_dnases( dnase, contacts[:,1], window, DATA_DIR + 'chr{}.full.x2dnase.npy'.format(chrom) )

	y = numpy.concatenate(( numpy.ones(positive.shape[0], dtype='float32'), numpy.zeros(negative.shape[0], dtype='float32') ))
	numpy.save( DATA_DIR + 'chr{}.full.y.npy'.format(chrom), y )

	os.system( 'mv {}chr{}* /net/noble/vol1/home/jmschr/proj/contact/data/'.format(DATA_DIR, chrom) )
