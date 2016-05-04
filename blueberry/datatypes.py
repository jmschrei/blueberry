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
import os
import pyximport

os.environ['CFLAGS'] = ' -I' + numpy.get_include()
pyximport.install()

from blueberry import *

RAO = "/net/noble/vol1/data/hic-datasets/results/Rao-Cell2014/"
DATA_DIR = RAO + "fixedWindowSize/fithic/afterICE/{0}/{2}.chr{1}.spline_pass1.res{0}.significances.txt.gz"
RAW_DIR = RAO + "rawFromGEO/{0}/{2}kb_resolution_intrachromosomal/chr{1}/MAPQGE30/chr{1}_{2}kb.KRnorm"

class RawContactMap(object):
	"""This is a contact map which has not been processed at all.

	This will convert a text file to a dense representation of the underlying
	contact map as a numpy array.

	Parameters
	----------
	celltype : str, options = (GM12878_combined, K562, IMR90, NHEK, HMEC, HUVEC) 
		The celltype to use.

	chromosome : int, range = (1, 22)
		The chromosome number to use.

	resolution : int
		The resolution of the hic experiment.
	"""

	def __init__(self, celltype, chromosome, resolution=1000):
		self.resolution = resolution
		self.filename = DATA_DIR.format(resolution, chromosome, celltype)
		self.chromosome = chromosome
		self.celltype = celltype

		data = pandas.read_csv(self.filename, sep="\t", engine='c', dtype='float64').values
		self.regions = numpy.union1d(self.map[:,0], self.map[:,1])

		n_bins = self.regions.max() / resolution
		self.matrix = numpy.zeros((n_bins, n_bins), dtype='float32')

		for i, j, contactCount in data.iterrow():
			self.matrix[i / resolution, j / resolution] = contactCount

	


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
		self.filename = DATA_DIR.format(resolution, chromosome, celltype)
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


