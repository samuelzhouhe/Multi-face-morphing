# -*- coding: utf-8 -*-
import numpy as np
import requests
import sys
import urllib2
import urllib
import time
import json


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
	imgPath = '../../dataSet/test_' + str(imgindex)+'.png'



	http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
	key = "***"
	secret = "***"
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