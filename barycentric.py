'''
Using Barycentric Interpolation to generate morphs between two faces
Copyright: Zhou HE (zhead@connect.ust.hk)
'''

import numpy as np 	
import cv2
from scipy.spatial import Delaunay
import sys
from helperFuncs import extractLandmarks
from helperFuncs import landmarksFromFacepp
from helperFuncs import convertRGB
import copy
import matplotlib.pyplot as plt


# img0index = int(sys.argv[1])
# img1index = int(sys.argv[2])

# generate halfway image OF A USER-SPECIFIED ALPHA VALUE by barycentric interpolation
def generateMorphBarycentric(img0index, img1index, alphaValue):
	faceImg0 = cv2.imread('./dataSet/test_' + str(img0index)+'.png')
	faceImg1 = cv2.imread('./dataSet/test_' + str(img1index)+'.png')
	halfwayImage = np.zeros(faceImg0.shape)
	#alpha-independent invariants
	face0Landmarks = extractLandmarks(img0index)
	face1Landmarks = extractLandmarks(img1index)

	face0forDrawing = copy.deepcopy(faceImg0)
	face1forDrawing = copy.deepcopy(faceImg1)


	finalImg = copy.deepcopy(faceImg0)
	# for i in range(face0Landmarks.shape[0]):
	# 	lm = face0Landmarks[i]
	# 	cv2.circle(finalImg,(int(lm[0]), int(lm[1])), 2, (0,0,255), 2)


	#alphas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
	alphas = [alphaValue]

	returnImg = np.zeros(faceImg0.shape)
	for alpha in alphas:
		print ("Generating the morph of alpha=" +str(alpha) + "...")
		halfwayImage = np.zeros(faceImg0.shape)	#reset halfway image for every alpha



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
					p0 = halfwayLandmarks[v0]	# point 0
					p1 = halfwayLandmarks[v1]
					p2 = halfwayLandmarks[v2]


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

					# add color
					for ch in xrange(3):
						halfwayImage[y][x][ch] += faceImg0[img0y][img0x][ch] * (1-alpha)


					#go to image1, find THE corresponding pixel
					img1p0 = face1Landmarks[v0]
					img1p1 = face1Landmarks[v1]
					img1p2 = face1Landmarks[v2]

					img1coord = img1p0*barycentric[0][0] + img1p1*barycentric[0][1] + img1p2*barycentric[0][2]
					img1x = int(img1coord[0])
					img1y = int(img1coord[1])
					
					# add color
					for ch in xrange(3):
						halfwayImage[y][x][ch] += faceImg1[img1y][img1x][ch] * alpha

		# draw the halfway landmarks onto the newly-formed halfway image
		# for i in range(halfwayLandmarks.shape[0]):
		# 	lm = halfwayLandmarks[i]
		# 	cv2.circle(halfwayImage,(int(lm[0]), int(lm[1])), 2, (3*i,0,255), 2)

		returnImg = halfwayImage
		finalImg = np.concatenate((finalImg,halfwayImage),axis=1)


	# for i in range(face1Landmarks.shape[0]):
	# 	lm = face1Landmarks[i]
	# 	cv2.circle(faceImg1,(int(lm[0]), int(lm[1])), 2, (0,0,255), 2)


	finalImg = np.concatenate((finalImg,faceImg1),axis = 1)
	return returnImg


	# cv2.imwrite(str(img0index) + '-' + str(img1index) + '-facepp-Barycentric-10faces' + '.jpg',finalImg)

