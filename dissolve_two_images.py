import numpy as np 
import cv2

img0index = 0
img1index = 4

print('test_' + str(img0index)+'.png')

faceImg0 = cv2.imread('../dataSet/test_' + str(img0index)+'.png')
faceImg1 = cv2.imread('../dataSet/test_' + str(img1index)+'.png')
halfwayImage = np.zeros(faceImg0.shape)


with open ('./points.txt') as file:
	for i in range(img0index-1):
		file.readline()

	#Read landmarks of img1
	content = file.readline()
	landmarks = content.split()
	del landmarks[0]
	for i in xrange(len(landmarks)):
		landmarks[i] = float(landmarks[i])
	print(landmarks)
