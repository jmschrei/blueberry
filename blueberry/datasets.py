# datasets.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

import random
import numpy
import os
import pyximport
import time

os.environ['CFLAGS'] = ' -I' + numpy.get_include()
pyximport.install()

from blueberry import *
from joblib import Parallel, delayed

random.seed(0)
numpy.random.seed(0)

ALL_CHROM = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]

def unpack_dnase_hypersensitivity(chrom, verbose=True):
	"""Break up the DNase bedgraph file into chromosomes.

	This will read the bedgraph file and extract a single chromosome from the
	data, converting it from bedgraph into an appropriate numpy array.

	Parameters
	----------
	chrom : int
		The chromosome to process.

	verbose : bool, optional
		Whether to print out when it is starting or done with this file.
		Defaults to True.

	Returns
	-------
	None
	"""

	if verbose:
		print "Starting | unpack_dnase_hypersensitivity | chr{}".format(chrom)
	
	with open( DATA_DIR + "E116-DNase.fc.signal.bedgraph", "r" ) as infile:
		dnase = numpy.zeros(300000000)
		seen = False
		str_chrom = str(chrom)

		for line in infile:
			if line[3:5].strip() != str_chrom:
				if seen:
					break
				continue

			seen = True

			line = line.split()
			start, end = int(line[1]), int(line[2])
			dnase[start:end] = float(line[3])
 
		numpy.save( DATA_DIR + 'chr{}.dnase'.format(chrom), dnase[:end] )

	if verbose:
		print "Finished | unpack_dnase_hypersensitivity | chr{}".format(chrom)

def unpack_chromosomes(chrom, verbose=True):
	"""Convert FastA files of chromosomes into numpy arrays.

	This will read a FastA file of a given chromosome and one hot encoded it
	into a matrix of size (n_nucleotides, 4).

	Parameters
	----------
	chrom : int
		The chromosome to process.

	verbose : bool, optional
		Whether to print out when it is starting or done with this file.
		Defaults to True.

	Returns
	-------
	None
	"""

	if verbose:
		print "Starting | unpack_chromosomes | chr{}".format(chrom)

	mapping = { 'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3 }
	with open( DATA_DIR + 'chr{}.fa'.format( chrom ), 'r' ) as infile:
		chromosome = ''.join( line.upper().strip('\r\n') for line in infile if not line.startswith('>') )
	
	encoded_chromosome = translate( numpy.array( list(chromosome) ), mapping )
	numpy.save( DATA_DIR + 'chr{}.ohe'.format( chrom ), encoded_chromosome )

	if verbose:
		print "Finished | unpack_chromosomes | chr{}".format(chrom)

def unpack_contacts(chrom, verbose=True):
	"""Extract the contacts and a random set of non-contacting regions.

	Extract all contacts which are statistically significant according to
	Fit-Hi-C, and a random selection of other regions. We define a 
	statistically significant contact to be one with a q-value less than 0.01.
	However, we filter out all contacts between 0.01 and 0.50 as being
	ambiguous. We end up saving all positive contacts, all contacts in the
	middle range, and a set of non-contacting region pairs.

	Parameters
	----------
	chrom : int
		The chromosome to process.

	verbose : bool, optional
		Whether to print out when it is starting or done with this file.
		Defaults to True.

	Returns
	-------
	None
	"""

	if verbose:
		print "Starting | unpack_contacts | chr{}".format(chrom)

	filename = "../HiC/GM12878_combined.chr{}.spline_pass1.res1000.significances.txt.gz".format( chrom )
	prefix = "Report   | unpack_contacts | chr{} | "
	report = "read {} lines, {} positive contacts {} middle contacts and {} regions"

	n, i, j, k = 2000000, 0, 0, 0
	positive_contacts = numpy.zeros((n, 2), dtype='int32')
	middle_contacts = numpy.zeros((n, 2), dtype='int32')
	regions = {}

	# We have two steps to do:
	#	(1) Extract the positive contacts and the middle range contacts
	#       0.01 < q-val < 0.50 from the file
	#   (2) Extract an equal number of non-contacting regions as positive
	#       contacts, making sure it's not in either.
	#
	# Part (1):
	# ---------

	infile = gzip.open( filename )
	for line in infile:
		j += 1

		if verbose and j % 100000 == 0:
			print prefix + report.format(chrom, j, i, k, len(regions.keys()))

		chr1, mid1, chr2, mid2, contactCount, p, q = line.split()

		if chr1 != chr2:
			continue

		mid1 = int(mid1)
		mid2 = int(mid2)

		dist = numpy.abs(mid1 - mid2)
		if dist < 10000 or dist > 10000000:
			continue

		regions[mid1] = 1
		regions[mid2] = 1

		if float(q) <= 0.01:
			positive_contacts[i, 0] = mid1
			positive_contacts[i, 1] = mid2
			i += 1

		elif float(q) <= 0.5:
			middle_contacts[k, 0] = mid1
			middle_contacts[k, 1] = mid2
			k += 1

	positive_contacts = positive_contacts[:i]
	positive_map = contacts_to_hashmap(positive_contacts)

	middle_contacts = middle_contacts[:k]
	middle_map = contacts_to_hashmap(middle_contacts)

	regions = numpy.array(regions.keys())

	# Part (2):
	# ---------
	
	n, i = i, 0

	negative_contacts = numpy.zeros((n, 2), dtype='int32')
	negative_map = {}

	while i	< n:
		mid1, mid2 = numpy.random.choice(regions, 2)
		dist = numpy.abs(mid1 - mid2)
		if dist < 10000 or dist > 10000000:
			continue


		if positive_map.has_key( (mid1, mid2) ):
			continue

		elif middle_map.has_key( (mid1, mid2) ):
			continue

		elif negative_map.has_key( (mid1, mid2) ):
			continue

		negative_contacts[i, 0] = mid1
		negative_contacts[i, 1] = mid2
		negative_map[(mid1, mid2)] = 1
		negative_map[(mid2, mid1)] = 1

		i += 1

	numpy.save( '../data/chr{}.full.negative_contacts.npy'.format(chrom), negative_contacts )
	numpy.save( '../data/chr{}.full.positive_contacts.npy'.format(chrom), positive_contacts )
	numpy.save( '../data/chr{}.full.middle_contacts.npy'.format(chrom), middle_contacts )
	numpy.save( '../data/chr{}.full.regions.npy'.format(chrom), regions )

	if verbose:
		print "Finished | unpack_contacts | chr{}".format(chrom)

def save_chrom_dataset( chrom, window=500 ):
	"""Extract the raw data for a chromosome and save it.

	Extract the one-hot encoded DNA data and the DNase sensitivity from a set
	of contacts and save it to a numpy file.

	Parameters
	----------
	chrom : int
		The chromosome to extract.

	window : int, optional
		The window around each contact to extract, should be resolution/2.
		Defaults to 500 for 1kb resolution.

	Returns
	-------
	None
	"""

	print "-> Starting chromosome {}".format( chrom )
	x1seq, x1dnase, x2seq, x2dnase, y = extract_dataset( chrom, window=window )

	numpy.save( '../data/chr{}.x1seq'.format( chrom ), x1seq )
	numpy.save( '../data/chr{}.x2seq'.format( chrom ), x2seq )
	numpy.save( '../data/chr{}.x1dnase'.format( chrom ), x1dnase )
	numpy.save( '../data/chr{}.x2dnase'.format( chrom ), x2dnase )
	numpy.save( '../data/chr{}.y'.format( chrom ), y )
	print "<- Finished chromosome {}".format( chrom )

def save_all_chrom_datasets( num_jobs=1, window=500 ):
	"""Extract the raw data for all chromosomes and save them.

	Extract the entire genome, one chromosome at a time. This can be done in
	parallel but since this is I/O bound not CPU bound it may not be
	significantly faster.

	Parameters
	----------
	n_jobs : int, optional
		The number of jobs to run in parallel. Defaults to 1.

	window : int, optional
		The window around each contact to extract, should be resolution/2.
		Defaults to 500 for 1kb resolution.

	Returns
	-------
	None
	"""

	with Parallel( n_jobs=n_jobs ) as p:
		p( delayed( save_chrom_dataset )( chrom, window ) for chrom in ALL_CHROM )

def build_dataset( chroms, name='data', verbose=True, downsample=1 ):
	"""Build a dataset from a set of chromosomes.

	This will read in the stored data for each chromosome and combine it into a
	single dataset. This can be used to create arbitrary train/test splits from
	the original data. This will utilize the SSD for writing, which can be up to
	5x faster than writing to the HDD. Also, tech support complains when we write
	to the NFS because the number of packets goes above their complaint line of
	200/s to 20,000/s.

	Parameters
	----------
	chroms : array-like
		The IDs of the chromosome to use for this process.

	name : str, optional
		The name of the file to build. 

	verbose : bool, optional
		Whether to bring ocassional status reports and commands

	downsample : int, optional
		The amount of data to use. 1 means use all data, 2 means use half of
		the data, etc...
	"""

	DATA_DIR = '/data/scratch/ssd/jmschr/contact/'

	sizes = []
	for i in chroms:
		data = numpy.load( '../data/chr{}.full.y.npy'.format(i), mmap_mode='r' )[::slice]
		sizes.append( int(data.shape[0] / downsample) )

	n = sum(sizes)
	idxs = numpy.cumsum([0] + sizes)

	outpaths = [ DATA_DIR + '{}.x1seq.npy'.format(name),
			     DATA_DIR + '{}.x2seq.npy'.format(name),
			     DATA_DIR + '{}.x1dnase.npy'.format(name),
			     DATA_DIR + '{}.x2dnase.npy'.format(name),
			     DATA_DIR + '{}.x1coord.npy'.format(name),
		     	 DATA_DIR + '{}.x2coord.npy'.format(name),
			     DATA_DIR + '{}.y.npy'.format(name)
	]

	x1seq   = numpy.memmap( outpaths[0], dtype='int8', mode='w+', shape=(n, 1, 1001, 4) )
	x2seq   = numpy.memmap( outpaths[1], dtype='int8', mode='w+', shape=(n, 1, 1001, 4) )
	x1dnase = numpy.memmap( outpaths[2], dtype='float32', mode='w+', shape=(n, 1, 1001, 1) )
	x2dnase = numpy.memmap( outpaths[3], dtype='float32', mode='w+', shape=(n, 1, 1001, 1) )
	x1coord = numpy.memmap( outpaths[4], dtype='int32', mode='w+', shape=(n,) )
	x2coord = numpy.memmap( outpaths[5], dtype='int32', mode='w+', shape=(n,) )
	y       = numpy.memmap( outpaths[6], dtype='float32', mode='w+', shape=(n,) )

	dbs = [x1seq, x2seq, x1dnase, x2dnase, x1coord, x2coord, y]
	names  = ['x1seq', 'x2seq', 'x1dnase', 'x2dnase', 'x1coord', 'x2coord',  'y']

	indexes = numpy.arange(n)
	numpy.random.shuffle(indexes)
	numpy.save( DATA_DIR + "{}.indexes.npy".format(name), indexes )

	step = 100000

	for db, name, p in zip(dbs, names, outpaths):
		j = 0
		for i, chrom in enumerate( chroms ):
			path = '../data/chr{}.full.{}.npy'.format(chrom, name)

			if 'dnase' in name:
				type = 'float32'
				data = numpy.memmap( path, dtype='float32', mode='r', shape=(sizes[i], 1, 1001, 1))
			elif 'seq' in name:
				type = 'int8'
				data = numpy.memmap( path, dtype='float32', mode='r', shape=(sizes[i], 1, 1001, 4))
			elif 'coord' in name:
				type = 'int32'
				data = numpy.load( path )
			elif 'y' in name:
				type = 'float32'
				data = numpy.load( path )

			k = data.shape[0]
			l = 0

			while k - l > step:
				tic = time.time()
				db[ indexes[j+l:j+l+step] ] = data[l:l+step]
				db.flush()

				if verbose:
					print "Incorporating samples {}-{} from {}, {}s".format(l, l+step, path, time.time() - tic)
				
				l += step

			tic = time.time()
			db[ indexes[j+l:j+k] ] = data[l:k]
			db.flush()
			
			if verbose:
				print "Incorporating samples {}-{} from {}, {}s".format(l, k, path, time.time() - tic)

			j += k

		if verbose:
			print 'mv {} ../data/'.format(p)
		
		os.system( 'mv {} ../data/'.format(p) )
		
		if verbose:
			print "chmod 0444 ../data/{}".format(p.split('/')[-1])
		
		os.system( 'chmod 0444 ../data/{}'.format(p.split('/')[-1]))
		
		if verbose:
			print "Done with chr{} {} in {}s".format(chrom, name, time.time() - tic)