import numpy as np
import requests


#helper function, which reads file and returns a list of 68 two-tuples.
def extractLandmarks(imgindex):
	with open ('./points.txt') as file:
		for i in range(imgindex-1):
			file.readline()
		
		#Read landmarks of img1
		content = file.readline()
		pointsInputList = content.split()
		del pointsInputList[0]
		landmarks = []
		for i in xrange(len(pointsInputList)/2):
			xcoord = int(2*float(pointsInputList[2*i]))
			ycoord = int(2*float(pointsInputList[2*i+1]))
			landmarks.append((xcoord,ycoord))
		
		return np.array(landmarks)

def getHorizontalSpan(p1,p2,p3):
	leftMost = min(p1[0],p2[0],p3[0])
	rightMost = max(p1[0],p2[0],p3[0])
	return rightMost - leftMost


def getVerticalSpan(p1,p2,p3):
	lowerMost = min(p1[1],p2[1],p3[1])
	upperMost = max(p1[1],p2[1],p3[1])
	return upperMost - lowerMost


def landmarksFromFacepp(imgindex):
	pass