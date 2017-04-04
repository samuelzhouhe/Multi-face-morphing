import numpy as np 	
import cv2
import sys
from scipy.spatial import Delaunay



img0index = int(sys.argv[1])
img1index = int(sys.argv[2])
#alpha = 0.25	#ranges from 0 to 1. the extend of morph


faceImg0 = cv2.imread('../../dataSet/test_' + str(img0index)+'.png')
faceImg1 = cv2.imread('../../dataSet/test_' + str(img1index)+'.png')
halfwayImage = np.zeros(faceImg0.shape)


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


finalImg = faceImg0
#alphas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
alphas = [0.5]
for alpha in alphas:
	print ("Generating the morph of alpha=" +str(alpha) + "...")

	halfwayImage = np.zeros(faceImg0.shape)	#reset
	face0Landmarks = extractLandmarks(img0index)
	face1Landmarks = extractLandmarks(img1index)
	halfwayLandmarks = face0Landmarks * (1-alpha) + face1Landmarks * alpha



	#triangulation on halfway image, based on halfway landmarks
	halfwayTri = Delaunay(halfwayLandmarks)


	#(x,y) is the pixel
	for x in range(faceImg0.shape[0]):
		for y in range(faceImg0.shape[1]):
			pixel = np.array([x,y])
			index = Delaunay.find_simplex(halfwayTri,pixel) #which triangle
			if index!= -1:
				for ch in xrange(3):
					halfwayImage[x][y][ch] += faceImg0[x][y][ch] * (1-alpha)
					halfwayImage[x][y][ch] += faceImg1[x][y][ch] * (alpha)

	finalImg = np.concatenate((finalImg,halfwayImage),axis=1)


finalImg = np.concatenate((finalImg,faceImg1),axis = 1)
cv2.imwrite(str(img0index) + '-' + str(img1index) + '-Naive-benchmark' + '.jpg',finalImg)





