
# coding: utf-8

# In[17]:

from helperFuncs import extractLandmarks
from helperFuncs import landmarksFromFacepp
from helperFuncs import getAvgColorDiff
from helperFuncs import getGlobalColorDiff
import numpy as np 
from sklearn import preprocessing
from tsp_solver_greedy import solve_tsp
import multiprocessing as mp



def computeAndStoreDifference(i, j,output):
	print("Calculate the diff between ", i, " and ", j)
	iLandmarks = extractLandmarks(i)
	jLandmarks = extractLandmarks(j)

	sumSquared = 0	#total pixel distortion
	for k in range(iLandmarks.shape[0]):
		distortion = iLandmarks[k] - jLandmarks[k]
		sumSquared += distortion[0] ** 2 + distortion[1] ** 2

	colorDiff = getGlobalColorDiff(i,j)

	output.put([sumSquared, colorDiff, i, j])
	return [sumSquared, colorDiff, i, j]
	


# there are 201 photos
def getOptimalSequence(seqSize,distortionWeight):
	distortionDiffs = np.zeros([seqSize,seqSize])
	colorDiffs = np.zeros([seqSize,seqSize])
	output = mp.Queue()
	processes = []
	for i in range(seqSize):
		for j in range(i+1,seqSize):
			processes.append(mp.Process(target=computeAndStoreDifference, args=(i,j,output)))


	for p in processes:
	    p.start()
	for p in processes:
	    p.join()

	results = [output.get() for p in processes]
	for res in results:
		i = res[2]
		j = res[3]
		distortion = res[0]
		colorDiff = res[1] 
		distortionDiffs[i][j] = distortion
		distortionDiffs[j][i] = distortion
		colorDiffs[i][j] = colorDiff
		colorDiffs[j][i] = colorDiff
		
	print("All Differences Ready. Now start to normalize and do TSP...")
	colorDiffs = preprocessing.normalize(colorDiffs)
	distortionDiffs = preprocessing.normalize(distortionDiffs)
	totalDiffs = (1-distortionWeight)*colorDiffs + distortionWeight*distortionDiffs

	path = solve_tsp(totalDiffs)
	return path



