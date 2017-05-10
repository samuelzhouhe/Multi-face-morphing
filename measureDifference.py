
# coding: utf-8

# In[17]:

from helperFuncs import extractLandmarks
from helperFuncs import landmarksFromFacepp
from helperFuncs import getAvgColorDiff
from helperFuncs import getGlobalColorDiff
import numpy as np 
from sklearn import preprocessing
from tsp_solver_greedy import solve_tsp

# there are 201 photos
def getOptimalSequence(endIndex):
	TESTSIZE = endIndex
	mostSimilarPair = [-1,-1]
	smallestWeightedDiff = 0
	distortionDiffs = np.zeros([TESTSIZE,TESTSIZE])
	colorDiffs = np.zeros([TESTSIZE,TESTSIZE])
	for i in range(TESTSIZE):
		for j in range(i+1,TESTSIZE):
			print("calculating " + str(i) + " " + str(j) + "...")
			iLandmarks = extractLandmarks(i)
			jLandmarks = extractLandmarks(j)

			sumSquared = 0	#total pixel distortion
			for k in range(iLandmarks.shape[0]):
				distortion = iLandmarks[k] - jLandmarks[k]
				sumSquared += distortion[0] ** 2 + distortion[1] ** 2

			colorDiff = getGlobalColorDiff(i,j)
			print(sumSquared, ' ', colorDiff)
			weightedDiff = float(sumSquared)/float(300) + colorDiff
			if i == 0 and j == 1:
				smallestWeightedDiff = weightedDiff
			elif weightedDiff < smallestWeightedDiff:
				smallestWeightedDiff = weightedDiff
				mostSimilarPair[0] = i
				mostSimilarPair[1] = j
			
			print("Total diff between img", i,  j, " " , " is ",weightedDiff)
			# distance[i][j] = weightedDiff
			distortionDiffs[i][j] = sumSquared
			distortionDiffs[j][i] = sumSquared
			colorDiffs[i][j] = colorDiff
			colorDiffs[j][i] = colorDiff


	# In[18]:

	colorDiffs = preprocessing.normalize(colorDiffs)

	distortionDiffs = preprocessing.normalize(distortionDiffs)

	totalDiffs = 0.1*colorDiffs + 0.9*distortionDiffs


	# In[19]:

	path = solve_tsp(totalDiffs)
	return path




