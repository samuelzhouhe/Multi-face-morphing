'''
Using Bilinear Interpolation to generate morphs between two faces
Copyright: Zhou HE (zhead@connect.ust.hk)
'''
import numpy as np 	
import cv2
from scipy.spatial import Delaunay
import sys
import matplotlib.pyplot as plt
from helperFuncs import extractLandmarks
from helperFuncs import getHorizontalSpan
from helperFuncs import getVerticalSpan


img0index = int(sys.argv[1])
img1index = int(sys.argv[2])
#alpha = 0.25	#ranges from 0 to 1. the extend of morph


faceImg0 = cv2.imread('./dataSet/test_' + str(img0index)+'.png')
faceImg1 = cv2.imread('./dataSet/test_' + str(img1index)+'.png')
halfwayImage = np.zeros(faceImg0.shape)




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
					halfwayImage[x][y][ch] +=  faceImg0[img0pixelX][img0pixelY][ch] * (1-alpha)



				#go to image1, find THE corresponding pixel
				img1p0 = face1Landmarks[v0]
				img1p1 = face1Landmarks[v1]
				img1p2 = face1Landmarks[v2]
				img1pixelX = int(min(img1p0[0],img1p1[0],img1p2[0]) + xOffsetPercent*getHorizontalSpan(img1p0,img1p1,img1p2))
				img1pixelY = int(min(img1p0[1],img1p1[1],img1p2[1]) + yOffsetPercent*getVerticalSpan(img1p0,img1p1,img1p2))
				img1pixelX = min(img1pixelX,511)
				img1pixelY = min(img1pixelY,511)

				for ch in range(3):
					halfwayImage[x][y][ch] +=  faceImg1[img1pixelX][img1pixelY][ch] * alpha
				#print(halfwayImage[x][y])

	finalImg = np.concatenate((finalImg,halfwayImage),axis=1)


finalImg = np.concatenate((finalImg,faceImg1),axis = 1)
cv2.imwrite(str(img0index) + '-' + str(img1index) + '-Bilinear' + '.jpg',finalImg)





