# DataTypes.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

"""
The datatypes involved in this project. Currently supports a ContactMap
which is the output from running Fit-Hi-C.
"""

import pandas
import gzip
import numpy
cimport numpy
import os
import pyximport
import math
import time
import scipy

os.environ['CFLAGS'] = ' -I' + numpy.get_include()
pyximport.install()

from blueberry import *

RAO = "/net/noble/vol1/data/hic-datasets/"
DATA_DIR = RAO + "results/Rao-Cell2014/fixedWindowSize/fithic/afterICE/{2}/{0}.chr{1}.spline_pass1.res{2}.significances.txt.gz"
RAW_DIR = RAO + "data/Rao-Cell2014/rawFromGEO/{0}/{2}kb_resolution_intrachromosomal/chr{1}/MAPQGE30/chr{1}_{2}kb.RAWobserved"
KR_NORM = RAO + "data/Rao-Cell2014/rawFromGEO/{0}/{2}kb_resolution_intrachromosomal/chr{1}/MAPQGE30/chr{1}_{2}kb.KRnorm"
KR_EXP = RAO + "data/Rao-Cell2014/rawFromGEO/{0}/{2}kb_resolution_intrachromosomal/chr{1}/MAPQGE30/chr{1}_{2}kb.KRexpected"

cdef class ContactMap(object):
	"""This is a contact map for Hi-C datasets.

	This takes in the contact map text file, formatted as the sparse
	representation of the upper triangle of the matrix in the format
	(i, j, contactCount) where contactCount is an integer of contacts recorded.
	In addition, it will load up the KR norm and the expected values
	from the same directory to use for matrix balancing and O/E normalization.
	It implements general methods for processing the matrix and plotting it.


	Parameters
	----------
	celltype : str, options = (GM12878_combined, K562, IMR90, NHEK, HMEC, HUVEC) 
		The celltype to use.

	chromosome : int, range = (1, 22)
		The chromosome number to use.

	resolution : int
		The resolution of the hic experiment.

	Attributes
	----------
	resolution : int
		The resolution of the stored hic experiment

	chromosome : int
		The chromosome of the stored hic experiment

	celltype : str
		The celltype of the stored hic experiment

	filename : str
		The filename of the raw count file used

	n_bins : int
		The number of bins found in matrix

	matrix : numpy.ndarray, shape=(n_bins+1, n_bins+1)
		The counts stored in the contact map. If the data is processed using
		some normalization or correlation method, the results are stored here.

	regions : numpy.ndarray, shape=(n_bins)
		The midpoints which were found in this map.
	"""

	cdef public int resolution
	cdef public int n_bins
	cdef public int chromosome
	cdef public str filename
	cdef public str celltype
	cdef numpy.ndarray KRnorm
	cdef numpy.ndarray KRexpected
	cdef public numpy.ndarray matrix
	cdef public numpy.ndarray regions 

	def __init__(self, celltype, chromosome, resolution=1000):
		self.resolution = resolution
		self.filename = RAW_DIR.format(celltype, chromosome, resolution/1000)
		self.chromosome = chromosome
		self.celltype = celltype

		self.KRnorm = numpy.loadtxt(KR_NORM.format(celltype, chromosome, resolution/1000))
		self.KRexpected = numpy.loadtxt(KR_EXP.format(celltype, chromosome, resolution/1000))
		self.n_bins = self.KRnorm.shape[0]
		cdef int d = self.n_bins+1

		cdef numpy.ndarray matrix = numpy.zeros((d, d), dtype='float64')
		cdef numpy.ndarray data = pandas.read_csv(self.filename, delimiter="\t", engine='c', 
												  dtype='float64', header=None).values
		data = numpy.nan_to_num(data)
		cdef int n = data.shape[0]

		cdef double* data_ptr = <double*> data.data
		cdef double* matrix_ptr = <double*> matrix.data
		cdef int i, j, k
		cdef double contactCount

		for i in range(n):
			j = data_ptr[i] / resolution
			k = data_ptr[n+i] / resolution
			contactCount = data_ptr[2*n+i]

			matrix_ptr[j*d + k] = contactCount
			matrix_ptr[k*d + j] = contactCount

		self.matrix = matrix
		self.regions = numpy.union1d(data[:,0], data[:,1])
		self.regions.sort()

	def filter(self, threshold=0):
		"""Remove rows and columns which are unmappable.

		Some regions of the genome are unmappable, and so no hic contacts will
		map to them. These rows/columns will have a marginal contact count of 0,
		and are not worth considering. This function removes rows and columns
		from this matrix which are less than a certain threshold.

		Parameters
		----------
		threshold : double
			The marginal count of a row/column which if below, causes removal 

		Returns
		-------
		None
		"""

		marginals = self.matrix.sum(axis=0)
		self.matrix = self.matrix[marginals > threshold][:, marginals > threshold]

	def normalize(self):
		"""Perform observed/expected normalization and matrix balancing.
		
		This will perform matrix balancing by dividing the count by the KR norm
		for that row and for that column, and then perform observed/expected
		normalization by dividing by the expected number of contacts for that
		genomic distance away from the diagonal. These operations are done
		directly on the underlying matrix.

		Parameters
		----------
		None

		Returns
		-------
		None
		"""

		cdef int i, j, d = self.n_bins+1
		cdef double* matrix = <double*> self.matrix.data
		cdef double* KRnorm = <double*> self.KRnorm.data
		cdef double* KRexpected = <double*> self.KRexpected.data

		for i in range(self.n_bins):
			for j in range(self.n_bins-i):
				matrix[j*d+j+i] /= KRnorm[j] * KRnorm[j+i] * KRexpected[i]
				matrix[(j+i)*d+j] = matrix[j*d+j+i]

		self.matrix = numpy.nan_to_num(self.matrix)

	def correlation(self):
		"""Convert the underlying map to a correlation map.

		This will convert the contact map to a correlation map, where all
		entries correspond to the correlation between row i and column j.

		Parameters
		----------
		None

		Returns
		-------
		None
		"""

		self.matrix = numpy.corrcoef(self.matrix)

	def plot(self, arcsinh=True, **kwargs):
		"""This will plot the contact map onto the current palette.

		Parameters
		----------
		arcsinh : bool
			Perform an hyperbolic inverse sin transformation of the data before
			plotting it. This is similar to a log transform, except it is defined
			at 0. Default is True.

		**kwargs : pyplot.imshow arguments
			These keyword arguments will be passed directly into matplotlibs
			imshow's plotting function.
		"""

		plt.title("{} chr{} at {}kb resolution".format(self.celltype, self.chromosome, self.resolution/1000), fontsize=16)
		plt.xlabel("Genomic Coordinate (kb)", fontsize=14)
		plt.ylabel("Genomic Coordinate (kb)", fontsize=14)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		
		if arcsinh:
			plt.imshow( numpy.arcsinh(self.matrix), **kwargs )
		else:
			plt.imshow( self.matrix, **kwargs )

	def eigenvector(self):
		"""Return the first eigenvector of the matrix.

		This uses the Restarted Lanczos Method as implemented by LAPLAC and
		wrapped by scipy to find only the first eigenvector and eigenvalue,
		without spending time calculating all of them. This can cause a
		significant speed improvement, since we only care about the first one.

		Parameters
		----------
		None

		Returns
		-------
		eigenvector : numpy.ndarray, shape=(n_bins,)
			The first eigenvector of the underlying matrix.
		"""

		eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(self.matrix, k=1)
		return eigenvectors[:,0]

	@classmethod
	def from_arrays(cls, contacts, KRnorm=None, KRexpected=None):
		"""Create a contact map from numpy arrays.

		Take in a sparse representation of contacts in the format of
		i, j, contactCount where i and j represent the mid points of contact,
		as well as optionally take in arrays corresponding to the KR matrix
		balancing norm and the expected number of contacts for a given
		genomic distance. No files required.

		Parameters
		----------
		contacts : numpy.ndarray, shape=(n, 3)
			Take in a sparse representation of the contacts in the format
			i, j, contactCount, where i and j are the two midpoints, and
			contactCount is the integer count of the number of contacts.

		KRnorm : numpy.ndarray
			The KR matrix balancing norm array. This contains a normalization
			constant for each column/row in the matrix. Optional for creating
			the object but required for normalization.

		KRexpected : numpy.ndarray
			The normalization constants for each genomic distance away from
			the diagonal. Optional for creating the object for required for
			normalization.
		"""

		self.KRnorm = KRnorm
		self.KRexpected = KRexpected
		self.regions = numpy.union1d(contacts[:,0], contacts[:,1])
		self.regions.sort()
		self.n_bins = self.regions.shape[0]

		cdef int d = self.n_bins + 1
		cdef numpy.ndarray matrix = numpy.zeros((d, d), dtype='float64')
		cdef numpy.ndarray data = numpy.nan_to_num(contacts).astype('float64')
		cdef int n = data.shape[0]

		cdef double* data_ptr = <double*> data.data
		cdef double* matrix_ptr = <double*> matrix.data
		cdef int i, j, k
		cdef double contactCount

		for i in range(n):
			j = data_ptr[i] / resolution
			k = data_ptr[n+i] / resolution
			contactCount = data_ptr[2*n+i]

			matrix_ptr[j*d + k] = contactCount
			matrix_ptr[k*d + j] = contactCount

		self.matrix = matrix

class FithicContactMap(object):
	"""This represents a contact map which has been processed with Fit-Hi-C.

	This will convert a text file to a sparse representation of the underlying
	contact map. Only q-values less than 1 are stored, as they are the minority
	and the q-value can be inferred from their absense. This also includes
	various operations which can be done on this map such as decimation to a
	lower resolution.

	Parameters
	----------
	celltype : str
		The celltype to use. (GM12878_combined, K562, IMR90, NHEK, HMEC, HUVEC)

	chromosome : int
		The chromosome number to use. (1-22)

	resolution : int
		The resolution of the hic experiment.

	Attributes
	----------
	filename : str or tuple
		The name of the file we are storing, either as a string, or as a tuple
		of (chr, resolution).

	regions : dict
		A dictionary storing all regions which have one or more contact (even
		if the contact has a q-value of 1)

	map : array-like, shape (n_contacts, 4)
		A numpy array storing (mid1, mid2, q, p) for each contact.
	"""

	def __init__(self, celltype, chromosome, resolution=1000):
		self.resolution = resolution
		self.filename = DATA_DIR.format(celltype, chromosome, resolution)
		self.chromosome = chromosome
		self.celltype = celltype

		self.map = pandas.read_csv(self.filename, sep="\t", usecols=[1, 3, 4, 5, 6], engine='c', dtype='float64').values
		self.regions = numpy.union1d(self.map[:,0], self.map[:,1])

	def decimate(self, resolution=5000):
		"""Decimate the map to a lower resolution using an aggregate.

		We do this by rounding the mid1 and mid2 values to the appropriate
		value and then aggregating by taking the product of the p-values.

		Parameters
		----------
		resolution : int
			The resolution we wish to decimate to.
		""" 

		self.resolution = resolution
		self.map[:,:2] = (self.map[:,:2].astype('int') + resolution) / resolution * resolution - resolution/2
		contact_values = {}

		for mid1, mid2, contactCount, p, q in self.map:
			key = mid1, mid2
			contact0, p0, q0 = contact_values.get(key, (0, 1, 1))
			contact_values[key] = contactCount + contact0, p*p0, min(q, q0)

		self.map = numpy.array([[mid1, mid2, contactCount, p, q] for (mid1, mid2), (contactCount, p, q) in contact_values.items()])
		self.regions = numpy.union1d(self.map[:,0], self.map[:,1])

	def contacts(self):
		"""Return all contacts with a q-value <= Q_LOWER_BOUND.

		Returns
		-------
		contacts : numpy.ndarray, shape=(n_contacts, 2)
			The pair of midpoints for each contact
		"""

		return self.map[self.map[:, 4] <= Q_LOWER_BOUND, :2]
