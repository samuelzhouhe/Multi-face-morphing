# -*- coding: utf-8 -*-
# from __future__ import division
import numpy as np
import requests
import sys
import urllib2
import urllib
import time
import json
import copy
import cv2
from scipy.spatial import Delaunay
import copy


def convertRGB(img):
	x = img.shape[0]
	y = img.shape[1]
	for i in range(x):
		for j in range(y):
			temp = img[i][j][0]
			img[i][j][0] = img[i][j][2]
			img[i][j][2] = temp
	return img




#helper function, which reads file and returns a list of 68 two-tuples.
def extractLandmarks(imgindex):
	with open ('./points.txt') as file:
		for i in range(imgindex):	#skip the first imgindex-1 lines. e.g. imgindex = 2 means skipping line0 and line1
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
	imgPath = './dataSet/test_' + str(imgindex)+'.png'



	http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
	key = "d3grtG6jlunxqfSbyVcSYaybSkSjPF6S"
	secret = "3ZPMJebhwWWijLgRnXfqDBR2rYf2Q3kw"
	filepath = imgPath
	boundary = '----------%s' % hex(int(time.time() * 1000))
	data = []
	data.append('--%s' % boundary)
	data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
	data.append(key)


	data.append('--%s' % boundary)
	data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
	data.append(secret)


	data.append('--%s' % boundary)
	fr=open(filepath,'rb')
	data.append('Content-Disposition: form-data; name="%s"; filename="co33.jpg"' % 'image_file')
	data.append('Content-Type: %s\r\n' % 'application/octet-stream')
	data.append(fr.read())
	fr.close()

	data.append('--%s' % boundary)
	data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
	data.append(str(1))

	data.append('--%s' % boundary)
	data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
	data.append('ethnicity,age,gender')


	data.append('--%s--\r\n' % boundary)

	http_body='\r\n'.join(data)
	#buld http request
	req=urllib2.Request(http_url)
	#header
	req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
	req.add_data(http_body)
	try:
		#post data to server
		resp = urllib2.urlopen(req, timeout=5)
		#get response
		qrcont=resp.read()
		#print (qrcont)
		res = json.loads(qrcont)
		majorFace = res['faces'][0]
		landmarks = majorFace['landmark']
		print(len(landmarks))
		returnLandmarks = []
		for k,v in landmarks.iteritems():
			#print (str(v['x']) + "  " + str(v['y']))
			returnLandmarks.append((v['x'],v['y']))

		return np.array(returnLandmarks)



	except urllib2.HTTPError as e:
	    print (e.read())

# get the color diff with their mid-halfway image
def getAvgColorDiff(img1, img2):
	img0index = img1
	img1index = img2


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
	alphas = [0.5]


	for alpha in alphas:
		halfwayImage = np.zeros(faceImg0.shape)	#reset halfway image for every alpha



		halfwayLandmarks = face0Landmarks * (1-alpha) + face1Landmarks * alpha
		#triangulation on halfway image, based on halfway landmarks
		halfwayTri = Delaunay(halfwayLandmarks)
		
		# plt.triplot(halfwayLandmarks[:,0],halfwayLandmarks[:,1],halfwayTri.simplices.copy())
		# plt.plot(halfwayLandmarks[:,0], halfwayLandmarks[:,1], 'o')
		# plt.show()



		#(x,y) is the pixel
		numValidPixels = 0	#number of pixels actually present in midway image
		totalColorDiff = 0
		for x in range(faceImg0.shape[0]):
			for y in range(faceImg0.shape[1]):
				pixel = np.array([x,y])
				index = Delaunay.find_simplex(halfwayTri,pixel) #which triangle
				if index!= -1:
					numValidPixels += 1

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

					blueDiff = abs(faceImg0[y][x][0] - halfwayImage[y][x][0]) + abs(faceImg1[y][x][0] - halfwayImage[y][x][0])
					greenDiff = abs(faceImg0[y][x][1] - halfwayImage[y][x][1]) + abs(faceImg1[y][x][1] - halfwayImage[y][x][1])
					redDiff = abs(faceImg0[y][x][2] - halfwayImage[y][x][2]) + abs(faceImg1[y][x][2] - halfwayImage[y][x][2])
					totalColorDiff += float(blueDiff+greenDiff+redDiff)/float(3)

		return float(totalColorDiff)/float(numValidPixels)


# get the bounded facial area of both images.
# then compute the average color of all PIXELS INSIDE THE BOUNDED AREA
# return the difference in squared error of pixel differences
def getGlobalColorDiff(img0, img1):

	print(img0, "   ",img1)

	faceImg0 = cv2.imread('./dataSet/test_' + str(img0)+'.png')
	faceImg1 = cv2.imread('./dataSet/test_' + str(img1)+'.png')

	#Average color within the bounding area
	img0triangulation = Delaunay(extractLandmarks(img0))
	img0validPixels = 0
	img0faceColor = [float(0),float(0),float(0)]
	for x in range(faceImg0.shape[0]):
		for y in range(faceImg0.shape[1]):
			pixel = np.array([x,y])
			index = Delaunay.find_simplex(img0triangulation,pixel) #which triangle
			if index!= -1:
				img0validPixels += 1
				img0faceColor[0] += faceImg0[x][y][0]
				img0faceColor[1] += faceImg0[x][y][1]
				img0faceColor[2] += faceImg0[x][y][2]
	for i in range(3):
		img0faceColor[i] /= float(img0validPixels)

	#Average color within the bounding area
	img1triangulation = Delaunay(extractLandmarks(img1))
	img1validPixels = 0
	img1faceColor = [float(0),float(0),float(0)]
	for x in range(faceImg1.shape[0]):
		for y in range(faceImg1.shape[1]):
			pixel = np.array([x,y])
			index = Delaunay.find_simplex(img1triangulation,pixel) #which triangle
			if index!= -1:
				img1validPixels += 1
				img1faceColor[0] += faceImg1[x][y][0]
				img1faceColor[1] += faceImg1[x][y][1]
				img1faceColor[2] += faceImg1[x][y][2]
	for i in range(3):
		img1faceColor[i] /= float(img1validPixels)

	return abs(img0faceColor[0] - img1faceColor[0]) ** 2 + abs(img0faceColor[1] - img1faceColor[1]) ** 2 + abs(img0faceColor[1] - img1faceColor[1]) ** 2


	# if faceImg0.shape != faceImg1.shape:
	# 	return -1
	# else:
	# 	totalColorDiff = 0
	# 	for x in range(faceImg0.shape[0]):
	# 		for y in range(faceImg0.shape[1]):
	# 				blueDiff = abs(faceImg0[y][x][0]-faceImg1[y][x][0])
	# 				greenDiff = abs(faceImg0[y][x][1]-faceImg1[y][x][1])
	# 				redDiff = abs(faceImg0[y][x][2]-faceImg1[y][x][2])
	# 				totalColorDiff += float(blueDiff+greenDiff+redDiff)/float(3)

	# 	numValidPixels = faceImg0.shape[0] * faceImg0.shape[1]
	# 	return float(totalColorDiff)/float(numValidPixels)

