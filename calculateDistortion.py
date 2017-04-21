from helperFuncs import extractLandmarks
from helperFuncs import landmarksFromFacepp
from helperFuncs import getAvgColorDiff

# there are 201 photos

mostSimilarPair = [-1,-1]
smallestWeightedDiff = 0
for i in range(10):
	for j in range(i+1,10):
		print("calculating " + str(i) + " " + str(j) + "...")
		iLandmarks = extractLandmarks(i)
		jLandmarks = extractLandmarks(j)

		sumSquared = 0	#total pixel distortion
		for k in range(iLandmarks.shape[0]):
			distortion = iLandmarks[k] - jLandmarks[k]
			sumSquared += distortion[0] ** 2 + distortion[1] ** 2


		weightedDiff = float(sumSquared)/float(300) + getAvgColorDiff(i,j)
		if i == 0 and j == 1:
			smallestWeightedDiff = weightedDiff
		elif weightedDiff < smallestWeightedDiff:
			smallestWeightedDiff = weightedDiff
			mostSimilarPair[0] = i
			mostSimilarPair[1] = j
		
		print("Total diff between img", i,  j, " " , " is ",weightedDiff)
print(mostSimilarPair)
