#!/usr/bin/env python

'''
Created on Jan 29, 2014
Modified 11/8/2016


Author     : Ferhat Ay
Modified By: Jacob Schreiber <jmschr@cs.washington.edu>
'''

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, gzip, bisect, itertools
import numpy, pandas, scipy.special as scsp

from optparse import OptionParser
from scipy.stats.mstats import mquantiles
from scipy.interpolate import UnivariateSpline
from sklearn.isotonic import IsotonicRegression

##### global variables shared by functions ######
# intra-chromosomal contacts in-range
possibleIntraInRangeCount=0 # count of all possible inter-chr fragment pairs
observedIntraInRangeCount=0
observedIntraInRangeSum=0
# intra-chromosomal contacts
possibleIntraAllCount=0 # count of all possible intra-chr fragment pairs
observedIntraAllCount=0
observedIntraAllSum=0
# inter-chromosomal contacts
possibleInterAllCount=0 # count of all possible inter-chr fragment pairs
observedInterAllCount=0
observedInterAllSum=0

baselineIntraChrProb=0	# 1.0/possibleIntraAllCount
interChrProb=0 #  1.0/possibleInterAllCount

minObservedGenomicDist=500000000 # some number bigger than the biggest chromosome length
maxObservedGenomicDist=0
maxPossibleGenomicDist=0

#distScaling just avoids overflow - but is necessary for large genomes
distScaling=10000.0

#########################

class FitHiC(object):
	"""Fit-Hi-C transformer object. 

	This object will transform Hi-C matrices by calculating the p and q-values
	for each bin.

	Parameters
	----------
	libname : str
		A prefix that is going to be used for output file names

	resolution : int
		Length of fixed-size genomic windows used to process the data.

	n_bins : int, optional
		The number of equal-occupancy bins to process the data. Default is 100

	n_passes : int, optional
		The number of passes after the initial spline fit. Default is 2.

	max_dist : int, optional
		The upper bound on the mid-range distances. Default is no limit.

	min_dist : int, optional
		The lower bound on the mid-range distances. Default is no limit.
	"""

	def __init__(self, libname, resolution, n_bins=100, n_passes=2, 
		max_dist=-1, min_dist=-1):
		self.libname = libname
		self.resolution = resolution
		self.n_bins = n_bins
		self.n_passes = n_passes
		self.max_dist = max_dist
		self.min_dist = min_dist

	def fit_transform(self, interactions, fragments, biases="none", verbose=False):
		"""Run the Fit-Hi-C algorithm on some data and calculate p and q values.

		Parameters
		----------
		interactions : str
			File containing the list of contact counts between fragment/window/
			meta-fragment pairs.

		fragments : str
			File containing the list of midpoints (or start indices) of the
			fragments/windows/meta-fragments for the reference genome.

		biases : str, optional
			File containing the biases calculated by ICE for each locus. Default
			is none.

		verbose : bool, optional
			Whether to print a report as the algorithm is running. Default is
			False.
		"""

		fithic(self.libname, self.resolution, self.n_bins, self.min_dist, 
			self.max_dist, self.n_passes, interactions, fragments, biases)

def main():
	### parse the command line arguments
	usage = "usage: %prog [options]"
	parser = OptionParser(usage=usage)
	parser.add_option("-f", "--fragments", dest="fragsfile",
			help="File containing the list of midpoints (or start indices) of the fragments/windows/meta-fragments for the reference genome." )
	parser.add_option("-i", "--interactions", dest="intersfile",
			help="File containing the list of contact counts between fragment/window/meta-fragment pairs.")
	parser.add_option("-t", "--biases", dest="biasfile",
					  help="OPTIONAL: biases calculated by ICE for each locus are read from BIASFILE")
	parser.add_option("-r", "--resolution", dest="resolution",type="int",
					  help="Length of fixed-size genomic windows used to process the data. E.x. 10000")
	parser.add_option("-l", "--lib", dest="libname",
					  help="OPTIONAL: A prefix (generally the name of the library) that is going to be used for output file names.")
	parser.add_option("-b", "--n_bins", dest="n_bins", type="int",
					  help="OPTIONAL: Number of equal-occupancy bins to process the data. Default is 100")
	parser.add_option("-p", "--passes", dest="n_passes",type="int",
					  help="OPTIONAL: Number of passes after the initial spline fit. DEFAULT is 2 (spline-1 and spline-2).")
	parser.add_option("-U", "--upperbound", dest="max_dist", type="int",
					  help="OPTIONAL: Upper bound on the mid-range distances. DEFAULT no limit.")
	parser.add_option("-L", "--lowerbound", dest="min_dist", type="int",
					  help="OPTIONAL: Lower bound on the mid-range distances. DEFAULT no limit.")
	parser.set_defaults(verbose=True, n_bins=100, min_dist=-1, max_dist=-1, n_passes=2, libname="", biasfile='none')
	(options, args) = parser.parse_args()
	if len(args) != 0:
		parser.error("incorrect number of arguments")

	n_bins = options.n_bins # 100 by default
	max_dist = options.max_dist # -1 by default, means no upper bound
	min_dist = options.min_dist # -1 by default, means no lower bound
	libname = options.libname
	n_passes = options.n_passes
	resolution = options.resolution

	interactions = options.intersfile
	frags = options.fragsfile
	bias = options.biasfile

	fithic(libname, resolution, n_bins, min_dist, max_dist, n_passes, interactions, frags, bias)

def fithic(libname, resolution, n_bins, min_dist, max_dist, n_passes, interactions, frags, biases):
	# read the mandatory inumpyut files -f and -i
	mainDic = generate_FragPairs(frags, resolution, min_dist, max_dist)

	biasDic = {}
	if biases != 'none':
		biasDic = read_bias_file(biases)

	# read contacts in sparse form
	mainDic = read_interactions(mainDic, interactions, min_dist, max_dist)

	### DO THE FIRST PASS ###
	# calculate priors using original fit-hic and plot with standard errors
	print("\n\t\tSPLINE FIT PASS 1 (spline-1) \n"),
	x, y, yerr = calculate_probabilities(mainDic, n_bins, resolution, min_dist, max_dist, libname+".fithic_pass1")

	# now fit spline to the data using power-law residual by improving it  <residualFactor> times
	splineXinit, splineYinit, splineResidual = fit_spline(mainDic, x, y, yerr, 
		interactions, libname+".spline_pass1", biasDic, resolution, min_dist, max_dist) 


	print("\nExecution of fit-hic completed successfully. \n\n"),
	return # from main


def read_bias_file(infilename):
	sys.stderr.write("\n\nReading ICE biases. \n")
	biases, discarded = {}, 0

	with gzip.open(infilename, 'r') as infile:
		for i, line in enumerate(infile):
			chr, mid, bias = line.rstrip().split()
			mid = int(mid)
			bias = float(bias)

			if bias < 0.5 or bias > 2:
				bias = -1
				discarded += 1

			if chr not in biases:
				biases[chr] = {}
			if mid not in biases[chr]:
				biases[chr][mid] = bias

	sys.stderr.write("Out of " + str(i+1) + " loci " +str(discarded) +" were discarded with biases not in range [0.5 2]\n\n" )
	return biases

def calculate_probabilities(mainDic, n_bins, resolution, min_dist, max_dist, filename):
	print("\nCalculating probability means and standard deviations by equal-occupancy binning of contact counts\n"),
	print("------------------------------------------------------------------------------------\n"),
	outfile = open(filename+'.res'+str(resolution)+'.txt', 'w')


	desiredPerBin = (observedIntraInRangeSum)/n_bins
	print("observed intra-chr read counts in range\t"+repr(observedIntraInRangeSum)+ ",\tdesired number of contacts per bin\t" +repr(desiredPerBin)+",\tnumber of bins\t"+repr(n_bins)+"\n"),

	# the following five lists will be the print outputs
	x = [] # avg genomic distances of bins
	y = [] # avg interaction probabilities of bins
	yerr = [] # stderrs of bins

	interactionTotalForBinTermination = 0
	n = 0 # bin counter so far
	totalInteractionCountSoFar = 0
	distsToGoInAbin = []
	binFull = 0

	for i in range(0, maxPossibleGenomicDist+1, resolution):
		totalInteractionCountSoFar += mainDic[i][1]
		if in_range_check(i, min_dist, max_dist)==False:
			continue

		# if one distance has more than necessary counts to fill a bin
		if mainDic[i][1] >= desiredPerBin: 
			distsToGoInAbin.append(i)
			interactionTotalForBinTermination=0
			binFull=1

		# if adding the next bin will fill the bin
		elif interactionTotalForBinTermination+mainDic[i][1] >= desiredPerBin:
			distsToGoInAbin.append(i)
			interactionTotalForBinTermination=0
			binFull=1
		# if adding the next bin will fill the bin
		else:
			distsToGoInAbin.append(i)
			interactionTotalForBinTermination+=mainDic[i][1]
		#
		if binFull==1:
			n_pairs, n_interactions, avg_dist = 0.0, 0.0, 0.0 
			se_p = 0.0 # for now I'm not worrying about error etc.
			n += 1
			
			if n < n_bins:
				desiredPerBin=1.0*(observedIntraInRangeSum-totalInteractionCountSoFar)/(n_bins-n)

			for b in distsToGoInAbin:
				n_pairs += mainDic[b][0]
				n_interactions += mainDic[b][1]
				avg_dist += 1.0 * mainDic[b][0] * (b/distScaling)
			
			meanProbabilityObsv= (n_interactions/n_pairs) / observedIntraInRangeSum
			avg_dist = distScaling * (avg_dist/n_pairs)

			x.append(avg_dist)
			y.append(meanProbabilityObsv)
			yerr.append(se_p)
			
			interactionTotalForBinTermination = 0
			binFull = 0
			distsToGoInAbin = []

	return x, y, yerr

def	read_interactions(mainDic, infile, min_dist, max_dist):
	print("\nReading all the contact counts\n"),
	print("------------------------------------------------------------------------------------\n"),

	global observedInterAllSum 
	global observedInterAllCount
	global observedIntraAllSum
	global observedIntraAllCount
	global observedIntraInRangeSum
	global observedIntraInRangeCount
	global minObservedGenomicDist
	global maxObservedGenomicDist

	data = pandas.read_csv(infile, header=None, names=['chr1', 'mid1', 'chr2', 'mid2', 'contactCount'], sep='\t')

	for i, chr1, mid1, chr2, mid2, contactCount in data.itertuples():
		distance = int(mid2) - int(mid1)

		if chr1 != chr2:
			observedInterAllSum += contactCount
			observedInterAllCount += 1
		else:
			observedIntraAllSum += contactCount
			observedIntraAllCount += 1

		if (min_dist == -1 or (min_dist > -1 and distance > min_dist)) and\
			(max_dist == -1 or (max_dist > -1 and distance <= max_dist)):
			minObservedGenomicDist = min(minObservedGenomicDist, distance)
			maxObservedGenomicDist = max(maxObservedGenomicDist, distance)
			if distance in mainDic:
				mainDic[distance][1] += contactCount
			observedIntraInRangeSum += contactCount
			observedIntraInRangeCount +=1	

	print("Observed, Intra-chr in range: pairs= "+str(observedIntraInRangeCount) +"\t totalCount= "+str(observedIntraInRangeSum))
	print("Observed, Intra-chr all: pairs= "+str(observedIntraAllCount) +"\t totalCount= "+str(observedIntraAllSum))
	print("Observed, Inter-chr all: pairs= "+str(observedInterAllCount) +"\t totalCount= "+str(observedInterAllSum))
	print("Range of observed genomic distances [%d %d]" % (minObservedGenomicDist,maxObservedGenomicDist) + "\n"),
	return mainDic

def generate_FragPairs(infilename, resolution, min_dist, max_dist):
	print("\nEnumerating all possible intra-chromosomal fragment pairs in-range\n"),
	print("------------------------------------------------------------------------------------\n"),
	global maxPossibleGenomicDist
	global possibleIntraAllCount
	global possibleInterAllCount
	global possibleIntraInRangeCount
	global interChrProb
	global baselineIntraChrProb

	mainDic, allFragsDic = {}, {}
	infile = gzip.open(infilename, 'r')

	for i, line in enumerate(infile):
		chr, mid = line.split()[:2]
		if chr not in allFragsDic:
			allFragsDic[chr] = {}
		allFragsDic[chr][mid] = i

	infile.close()

	n_frags = 0
	maxFrags = {}
	for ch in allFragsDic:
		maxFrags[ch] = max([int(i)-resolution/2 for i in allFragsDic[ch]])
		n_frags += len(allFragsDic[ch])
		maxPossibleGenomicDist = max(maxPossibleGenomicDist,maxFrags[ch])

	for i in range(0, maxPossibleGenomicDist+1, resolution):
		mainDic[i] = [0,0]

	for ch in allFragsDic:
		maxFrag = maxFrags[ch]
		n = len(allFragsDic[ch])
		d = 0
		for i in range(0, maxFrag+1, resolution):
			mainDic[i][0] += n-d
			d += 1
		
		possibleInterAllCount += n * (n_frags-n)
		possibleIntraAllCount += (n*(n+1))/2 # n(n-1) if excluding self
	
	possibleInterAllCount /= 2
	interChrProb = 1.0 / possibleInterAllCount
	baselineIntraChrProb = 1.0 / possibleIntraAllCount
	
	for i in range(0, maxPossibleGenomicDist+1, resolution):
		if in_range_check(i, min_dist, max_dist):
			possibleIntraInRangeCount += mainDic[i][0]

	print("Number of all fragments= "+str(n_frags)+"\t resolution= "+ str(resolution))
	print("Possible, Intra-chr in range: pairs= "+str(possibleIntraInRangeCount))
	print("Possible, Intra-chr all: pairs= "+str(possibleIntraAllCount)) 
	print("Possible, Inter-chr all: pairs= "+str(possibleInterAllCount))
	print("Desired genomic distance range	[%d %d]" % (min_dist,max_dist) + "\n"),
	print("Range of possible genomic distances	[0	%d]" % (maxPossibleGenomicDist) + "\n"),

	return mainDic

def fit_spline(mainDic, x, y, yerr, infilename, outfilename, biasDic, resolution, min_dist, max_dist):
	print("\nFit a univariate spline to the probability means\n"),
	print("------------------------------------------------------------------------------------\n"),

	# maximum residual allowed for spline is set to min(y)^2
	splineError = min(y)**2

	# use fitpack2 method -fit on the real x and y from equal occupancy binning
	ius = UnivariateSpline(x, y, s=splineError)

	#### POST-PROCESS THE SPLINE TO MAKE SURE IT'S NON-INCREASING
	### NOW I DO THIS BY CALLING A SKLEARN ISOTONIC REGRESSION
	### This does the isotonic regression using option antitonic to make sure
	### I get monotonically decreasing probabilites with increasion genomic distance

	min_x, max_x = min(x), max(x)
	tempList=sorted([dis for dis in mainDic])
	splineX=[]
	### The below for loop will make sure nothing is out of range of [min(x) max(x)]
	### Therefore everything will be within the range where the spline is defined
	for i in tempList:
		if min_x <= i <= max_x:
			splineX.append(i)

	splineY=ius(splineX)
	
	ir = IsotonicRegression(increasing=False)
	rNewSplineY = ir.fit_transform(splineX, splineY)

	newSplineY=[]
	diff=[]
	diffX=[]
	for i in range(len(rNewSplineY)):
		newSplineY.append(rNewSplineY[i])
		if (splineY[i]-newSplineY[i]) > 0:
			diff.append(splineY[i]-newSplineY[i])
			diffX.append(splineX[i])

	### Now newSplineY holds the monotonic contact probabilities
	residual = sum([i*i for i in (y - ius(x))])

	### Now plot the results
	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(2,1,1)
	plt.title('Univariate spline fit to the output of equal occupancy binning. \n Residual= %e' % (residual),size='small')
	plt.plot([i/1000.0 for i in x], [i*100000 for i in y], 'ro', label="Means")
	plt.plot([i/1000.0 for i in splineX], [i*100000 for i in newSplineY],'g-',label="Spline fit")

	plt.ylabel('Probability (1e-5)')
	plt.xlabel('Genomic distance (kb)')
	plt.xlim([min_x/1000.0, max_x/1000.0])
	ax.legend(loc="upper right")

	ax = fig.add_subplot(2,1,2)
	plt.loglog(splineX,newSplineY,'g-')
	plt.loglog(x, y, 'r.')  # Data

	plt.ylabel('Probability (log scale)')
	plt.xlabel('Genomic distance (log scale)')
	plt.xlim([min_x, max_x])
	plt.savefig(outfilename+'.res'+str(resolution)+'.png')
	sys.stderr.write("Plotting %s" % outfilename + ".png\n")

	# NOW write the calculated pvalues and corrected pvalues in a file
	intraInRangeCount = 0
	intraOutOfRangeCount = 0
	intraVeryProximalCount = 0
	interCount = 0
	discardCount = 0

	print("lower bound on mid-range distances  "+ repr(min_dist) + ", upper bound on mid-range distances  " + repr(max_dist) +"\n"),
	p_vals=[]
	q_vals=[]

	with gzip.open(infilename, 'r') as infile:
		for line in infile:
			chr1, mid1, chr2, mid2, contactCount = line.rstrip().split()
			mid1 = int(mid1)
			mid2 = int(mid2)
			contactCount = int(contactCount)
			distance = mid2 - mid1

			
			bias1 = 1.0; 
			bias2 = 1.0;  # assumes there is no bias to begin with
			# if the biasDic is not null sets the real bias values
			if len(biasDic)>0:
				if chr1 in biasDic and mid1 in biasDic[chr1]:
					bias1=biasDic[chr1][mid1]
				if chr2 in biasDic and mid2 in biasDic[chr2]:
					bias2=biasDic[chr2][mid2]
		
			if bias1 == -1 or bias2 == -1:
				p_val = 1.0
				discardCount += 1
			elif chr1 == chr2:
				if (min_dist==-1 or (min_dist>-1 and distance >min_dist)) and\
				   (max_dist==-1 or (max_dist>-1 and distance <= max_dist)):
					# make sure the interaction distance is covered by the probability bins
					distToLookUp = min(max(distance, min_x), max_x)
					i = min(bisect.bisect_left(splineX, distToLookUp),len(splineX)-1)
					prior_p = newSplineY[i] * (bias1 * bias2) # biases added in the picture
					p_val = scsp.bdtrc(contactCount-1, observedIntraInRangeSum, prior_p)
					intraInRangeCount +=1
				elif (min_dist > -1 and distance <= min_dist):
					prior_p = 1.0
					p_val = 1.0
					intraVeryProximalCount += 1
				elif (max_dist>-1 and distance > max_dist):
					## out of range distance
					## use the prior of the baseline intra-chr interaction probability
					prior_p = baselineIntraChrProb * (bias1 * bias2)  # biases added in the picture
					p_val = scsp.bdtrc(contactCount-1, observedIntraAllSum, prior_p)
					intraOutOfRangeCount += 1
			else: 
				prior_p = interChrProb*(bias1*bias2) # biases added in the picture
				############# THIS HAS TO BE interactionCount-1 ##################
				p_val = scsp.bdtrc(contactCount-1, observedInterAllSum, prior_p)
				interCount += 1
			
			p_vals.append(p_val)

	# Do the BH FDR correction
	q_vals = benjamini_hochberg_correction(p_vals, possibleInterAllCount+possibleIntraAllCount)

	print("Writing p-values and q-values to file %s" % outfilename + ".significances.txt\n"),
	print("Number of pairs discarded due to bias not in range [0.5 2]\n"),

	with gzip.open(infilename, 'r') as infile:
		with gzip.open(outfilename+'.res'+str(resolution)+'.significances.txt.gz', 'w') as outfile:
			outfile.write("chr1\tfragmentMid1\tchr2\tfragmentMid2\tcontactCount\tp-value\tq-value\n")
			for i, line in enumerate(infile):
				p_val, q_val = p_vals[i], q_vals[i]
				chr1, mid1, chr2, mid2, contactCount = line.rstrip().split()
				outfile.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(chr1, mid1, chr2, mid2, contactCount, p_val, q_val))

	return splineX, newSplineY, residual



################### FUNC in_range_check  #####################################
####  Check whether the given interactionDistance is within the range we are 
###   interested. Should only be used for intra-chromosomal interactions.
##############################################################################
def in_range_check(interactionDistance, min_dist, max_dist):
	if (min_dist==-1 or (min_dist>-1 and interactionDistance >min_dist)) and\
		(max_dist==-1 or (max_dist>-1 and interactionDistance <= max_dist)):
		return True
	return False


################### FUNC benjamini_hochberg_correction  #####################
#### Given an array of p-values (not necessarily sorted) and the number of total 
### tests that were performed to gather these p-values, this function performs
###  the multiple hypothesis testing correction described by Benjamini-Hochberg.
###
### 
### If the number of tests are much more compared to p-value array and
### the omitted p-values are all supposed to be zero you should use it like:
### q_array=benjamini_hochberg_correction([0.03,0.4,0.7,0.01],10)
### 
### If the number of tests are equal to the ones in p-values array then:
### p_array=[0.03,0.4,0.7,0.01]
### q_array=benjamini_hochberg_correction(p_array,len(p_array))
##############################################################################
def benjamini_hochberg_correction(p_values, num_total_tests):
	# assumes that p_values vector might not be sorted
	pvalsArray = numpy.array(p_values)
	order = pvalsArray.argsort()
	sorted_pvals = numpy.take(p_values,order)
	q_values = [1.0 for i in range(len(p_values))]
	prev_bh_value = 0
	for i, p_value in enumerate(sorted_pvals):
		bh_value = p_value * num_total_tests / (i + 1)
		# Sometimes this correction can give values greater than 1,
		# so we set those values at 1
		bh_value = min(bh_value, 1)

		# To preserve monotonicity in the values, we take the
		# maximum of the previous value or this one, so that we
		# don't yield a value less than the previous.
		bh_value = max(bh_value, prev_bh_value)
		prev_bh_value = bh_value
		qIndex = order[i]
		q_values[qIndex] = bh_value

	return q_values

if __name__ == "__main__":
	main()

