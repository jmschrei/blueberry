# DataTypes.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

"""
The datatypes involved in this project. Currently supports a ContactMap
which is the output from running Fit-Hi-C.
"""

import pandas as pd
import gzip
import numpy
import os
import pyximport

os.environ['CFLAGS'] = ' -I' + numpy.get_include()
pyximport.install()

from blueberry import *

DATA_DIR = "/net/noble/vol1/data/hic-datasets/results/Rao-Cell2014/fixedWindowSize/fithic/afterICE/{0}/GM12878_combined.chr{1}.spline_pass1.res{0}.significances.txt.gz"

class ContactMap(object):
	"""This represents a contact map which has been processed with Fit-Hi-C.

	This will convert a text file to a sparse representation of the underlying
	contact map. Only q-values less than 1 are stored, as they are the minority
	and the q-value can be inferred from their absense. This also includes
	various operations which can be done on this map such as decimation to a
	lower resolution.

	Parameters
	----------
	filename : str
		The path to the file we wish to load.

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

	contacts : dict
		A dictionary with keys as (mid1, mid2) and values as (q, p).
	"""

	def __init__(self, filename):
		if isinstance( filename, tuple ):
			self.filename = DATA_DIR.format( *filename ) 

		self.filename = filename
		self.regions = {}
		self.map = []

		for line in gzip.open(filename):
			chr1, mid1, chr2, mid2, contactCount, p, q = line.split()

			if chr1 != chr2:
				continue

			mid1, mid2, contactCount, p, q = int(mid1), int(mid2), int(contactCount), float(p), float(q)

			self.regions[mid1] = 1
			self.regions[mid2] = 1

			if q == 1.0:
				continue

			self.map.append( [mid1, mid2, q, p] )

		self.map = numpy.array(self.map)
		self.regions = numpy.array(self.regions.keys(), dtype='int')
		self.contacts = contacts_to_qhashmap( self.map )

	def get(self, key, default):
		return self.contacts.get(key, default) 

	def has_contact(self, mid1, mid2):
		return self.contacts.has_key((mid1, mid2))

	def decimate(self, resolution=5000):
		"""Decimate the map to a lower resolution using an aggregate.

		We do this by rounding the mid1 and mid2 values to the appropriate
		value and then aggregating by taking the product of the p-values.

		Parameters
		----------
		resolution : int
			The resolution we wish to decimate to.
		""" 

		contacts = {}
		self.map[:,:2] = (self.map[:,:2].astype('int') + resolution) / resolution * resolution - resolution/2

		for mid1, mid2, q, p in self.map:
			key = mid1, mid2
			q0, p0 = contacts.get(key, (1, 1))

			q = q if q < q0 else q0
			contacts[key] = q, p*p0

		self.map = numpy.array([[mid1, mid2, q, p] for (mid1, mid2), (q, p) in contacts.items()])
		self.regions = numpy.unique( (self.regions + resolution) / resolution * resolution - resolution/2 )
		self.contacts = contacts_to_qhashmap( self.map )
