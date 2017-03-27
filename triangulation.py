import numpy as np 	
import cv2
from scipy.spatial import Delaunay
#import matplotlib.pyplot as plt

img0index = 37
img1index = 41


faceImg0 = cv2.imread('../dataSet/test_' + str(img0index)+'.png')
faceImg1 = cv2.imread('../dataSet/test_' + str(img1index)+'.png')
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





face0Landmarks = extractLandmarks(0)
face1Landmarks = extractLandmarks(4)
halfwayLandmarks = (face0Landmarks +face1Landmarks)/2

#plt.triplot(face0Landmarks[:,0],points[:,1],tri.simplices.copy())
#plt.plot(points[:,0], points[:,1], 'o')
#plt.show()

#triangulation on halfway image, based on halfway landmarks
halfwayTri = Delaunay(halfwayLandmarks)
print(halfwayTri.simplices)


def getHorizontalSpan(p1,p2,p3):
	leftMost = min(p1[0],p2[0],p3[0])
	rightMost = max(p1[0],p2[0],p3[0])
	return rightMost - leftMost


def getVerticalSpan(p1,p2,p3):
	lowerMost = min(p1[1],p2[1],p3[1])
	upperMost = max(p1[1],p2[1],p3[1])
	return upperMost - lowerMost

#sample a 50*50 region first.
#(x,y) is the pixel
for x in range(faceImg0.shape[0]):
	for y in range(faceImg0.shape[1]):
		index = Delaunay.find_simplex(halfwayTri,np.array([x,y])) #which triangle
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

			#bilinear interpolation based on the first two points?
			xOffsetPercent = float(x-min(p0[0],p1[0],p2[0]))/ float(xSpan)
			yOffsetPercent = float(y - min(p0[1],p1[1],p2[1]))/float(ySpan)


			#go back to image0, determine THE corresponding PIXEL in img0
			img0p0 = face0Landmarks[v0]
			img0p1 = face0Landmarks[v1]
			img0p2 = face0Landmarks[v2]
			img0pixelX = int(min(img0p0[0],img0p1[0],img0p2[0]) + xOffsetPercent*getHorizontalSpan(img0p0,img0p1,img0p2))
			img0pixelX = min(img0pixelX,511)
			img0pixelY = int(min(img0p0[1],img0p1[1],img0p2[1]) + yOffsetPercent*getVerticalSpan(img0p0,img0p1,img0p2))
			img0pixelY = min(img0pixelY,511)

			for ch in range(3):
				halfwayImage[x][y][ch] +=  faceImg0[img0pixelX][img0pixelY][ch] * 0.5



			#go to image1, find THE corresponding pixel
			img1p0 = face1Landmarks[v0]
			img1p1 = face1Landmarks[v1]
			img1p2 = face1Landmarks[v2]
			img1pixelX = int(min(img1p0[0],img1p1[0],img1p2[0]) + xOffsetPercent*getHorizontalSpan(img1p0,img1p1,img1p2))
			img1pixelY = int(min(img1p0[1],img1p1[1],img1p2[1]) + yOffsetPercent*getVerticalSpan(img1p0,img1p1,img1p2))
			img1pixelX = min(img1pixelX,511)
			img1pixelY = min(img1pixelY,511)

			for ch in range(3):
				halfwayImage[x][y][ch] +=  faceImg1[img1pixelX][img1pixelY][ch] * 0.5
			#print(halfwayImage[x][y])

finalImg = np.concatenate((faceImg0,halfwayImage,faceImg1),axis=1)

cv2.imwrite('TriangulatedMorph' + str(img0index) + '-' + str(img1index) +'.jpg',finalImg)





