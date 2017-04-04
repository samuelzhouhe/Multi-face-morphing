import numpy as np 	
import cv2
from scipy.spatial import Delaunay
import sys
import matplotlib.pyplot as plt

from scipy.interpolate import BarycentricInterpolator

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
	# plt.triplot(halfwayLandmarks[:,0],halfwayLandmarks[:,1],halfwayTri.simplices.copy())
	# plt.plot(halfwayLandmarks[:,0], halfwayLandmarks[:,1], 'o')
	# plt.show()







	#(x,y) is the pixel
	for x in range(faceImg0.shape[0]):
		for y in range(faceImg0.shape[1]):
			pixel = np.array([x,y])
			index = Delaunay.find_simplex(halfwayTri,pixel) #which triangle
			if index!= -1:

				#the indices of the vertices. not the points themselves
				v0 = halfwayTri.simplices[index][0]
				v1 = halfwayTri.simplices[index][1]
				v2 = halfwayTri.simplices[index][2]

				#The actual vertices of this triangle on halfway image
				p0 = halfwayLandmarks[v0]
				p1 = halfwayLandmarks[v1]
				p2 = halfwayLandmarks[v2]
				xSpan = getHorizontalSpan(p0,p1,p2)
				ySpan = getVerticalSpan(p0,p1,p2)

				#barycentric interpolation
				#determine the barycentric coordinates with respect to the three vertices of this simplex
				p = np.array([(x,y)])
				b = halfwayTri.transform[index,:2].dot(np.transpose(p - halfwayTri.transform[index,2]))
				barycentric = np.c_[np.transpose(b), 1-b.sum(axis=0)]


				#go back to image0, determine THE corresponding PIXEL in img0
				#use Barycentric here
				img0p0 = face0Landmarks[v0]
				img0p1 = face0Landmarks[v1]
				img0p2 = face0Landmarks[v2]

				img0coord = img0p0*barycentric[0][0] + img0p1*barycentric[0][1] + img0p2*barycentric[0][2]
				img0x = int(img0coord[0])
				img0y = int(img0coord[1])

				for ch in xrange(3):
					halfwayImage[x][y][ch] += faceImg0[img0x][img0y][ch] * (1-alpha)



				#go to image1, find THE corresponding pixel
				img1p0 = face1Landmarks[v0]
				img1p1 = face1Landmarks[v1]
				img1p2 = face1Landmarks[v2]

				img1coord = img1p0*barycentric[0][0] + img1p1*barycentric[0][1] + img1p2*barycentric[0][2]
				img1x = int(img1coord[0])
				img1y = int(img1coord[1])

				for ch in xrange(3):
					halfwayImage[x][y][ch] += faceImg1[img1x][img1y][ch] * alpha






	finalImg = np.concatenate((finalImg,halfwayImage),axis=1)


finalImg = np.concatenate((finalImg,faceImg1),axis = 1)
cv2.imwrite(str(img0index) + '-' + str(img1index) + '-Barycentric' + '.jpg',finalImg)




