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

from .blueberry import *
from joblib import Parallel, delayed

random.seed(0)
numpy.random.seed(0)

ALL_CHROM = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]

def unpack_dnase_hypersensitivity(celltype, chrom, verbose=True):
	"""Break up the DNase bedgraph file into chromosomes.

	This will read the bedgraph file and extract a single chromosome from the
	data, converting it from bedgraph into an appropriate numpy array.

	Parameters
	----------
	celltype : str
		The cell type to process.

	chrom : int
		The chromosome to process 

	verbose : bool, optional
		Whether to print out when it is starting or done with this file.
		Defaults to True.

	Returns
	-------
	None
	"""

	if verbose:
		print "Starting | unpack_dnase_hypersensitivity | chr{}".format(chrom)
	
	data = unpack_bedgraph(DATA_DIR + '/' + celltype + ".bedgraph", chromosome)
	numpy.save( DATA_DIR + '/chr{}.{}.dnase'.format(chrom, celltype), dnase[:end] )

	if verbose:
		print "Finished | unpack_dnase_hypersensitivity | chr{}".format(chrom)

def unpack_bedgraph(filename, chromosome):
	"""Read a bedgraph file and return a numpy array.

	Parameters
	----------
	filename : str
		The bedgraph file to open

	chromosome : int
		The chromosome to extract data for

	Returns
	-------
	data : numpy.ndarray
		The extracted data for a single chromosome
	"""

	n = numpy.load('../data/chr{}.ohe.npy'.format(chromosome), mmap_mode='r').shape[0]
	data = numpy.zeros(n)
	chromosome = str(chromosome)

	with open(filename, 'r') as infile:
		line = infile.readline()
		while line[3:5].strip() != chromosome:
			line = infile.readline()

		while line[3:5].strip() == chromosome:
			line = line.split()
			start, end = int(line[1]), int(line[2])
			data[start:end] = float(line[3])
			line = infile.readline()

	return data

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
