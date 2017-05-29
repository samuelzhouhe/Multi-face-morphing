
# coding: utf-8

# In[27]:

from barycentric import generateMorphBarycentric
from measureDifference import getOptimalSequence
import cv2
import numpy as np
import os
import multiprocessing as mp
from multiprocessing import Process
import sys

distortionWeight = float(sys.argv[1])	#weight of distortion. should be between 0.0 to 1.0
endIndex = int(sys.argv[2])	#index of the last image used in the project.

print(str(distortionWeight))

def generateAndWrite(ind1,ind2, alpha,count):
	serialNum = '0' * (5-len(str(count))) + str(count)
	print 'process id:', os.getpid()
	halfway = generateMorphBarycentric(ind1,ind2, alpha)
	print("Image generation finished for ", ind1, " ", ind2, " ", alpha)
	cv2.imwrite("./imgOutput" + str(distortionWeight) + "/" + serialNum + "Distort" + str(distortionWeight) + "-" +str(ind1) + "-" + str(ind2) + "-" + "Alpha-" + str(alpha) + ".jpg", halfway)
	print("Writing finished for ", ind1, " ", ind2, " ", alpha)
	



seq = getOptimalSequence(endIndex,distortionWeight)


print(seq)
morphJobs = []
for i in range(len(seq)):
	if i != len(seq)-1:
		for alpha in np.arange(0.0,1.00,0.04):
			morphJobs.append([seq[i],seq[i+1],alpha,len(morphJobs)])
		# finalImg = np.concatenate((finalImg,halfway),axis=1)
		# rhsImg = cv2.imread('./dataSet/test_' + str(seq[i+1])+'.png')
		# finalImg = np.concatenate((finalImg,rhsImg),axis=1)

processes = [mp.Process(target=generateAndWrite, args=(job[0],job[1],job[2],job[3])) for job in morphJobs]
for p in processes:
    p.start()
for p in processes:
    p.join()

