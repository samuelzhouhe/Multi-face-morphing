
# coding: utf-8

# In[27]:

from barycentric import generateMorphBarycentric
from measureDifference import getOptimalSequence
import cv2
import numpy as np
import os
import multiprocessing as mp
from multiprocessing import Process


def generateAndWrite(ind1,ind2, alpha):

	print 'process id:', os.getpid()
	halfway = generateMorphBarycentric(ind1,ind2, alpha)
	print("Image generation finished for ", ind1, " ", ind2, " ", alpha)
	cv2.imwrite("./imgOutput/Distortion" + str(distortionWeight) + "-" +str(ind1) + "-" + str(ind2) + "-" + "Alpha-" + str(alpha) + ".jpg", halfway)
	print("Writing finished for ", ind1, " ", ind2, " ", alpha)
	

# print(mp.cpu_count())
# p1 = Process(target=getOptimalSequence, args=(15,))
# p2 = Process(target=getOptimalSequence, args=(15,))

# p1.start()
# p2.start()
# p1.join()
# p2.join()
distortionWeight = 1.0


endIndex = 20
seq = getOptimalSequence(endIndex,distortionWeight)
# seq = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]	#hard code for now
# finalImg = cv2.imread('./dataSet/test_' + str(seq[0])+'.png')	# append very first img


# In[28]:


print(seq)
morphJobs = []
for i in range(len(seq)):
	if i != len(seq)-1:
		for alpha in np.arange(0.0,1.04,0.04):
			# generateAndWrite(seq[i],seq[i+1], alpha)
			morphJobs.append([seq[i],seq[i+1],alpha])
		# finalImg = np.concatenate((finalImg,halfway),axis=1)
		# rhsImg = cv2.imread('./dataSet/test_' + str(seq[i+1])+'.png')
		# finalImg = np.concatenate((finalImg,rhsImg),axis=1)

processes = [mp.Process(target=generateAndWrite, args=(job[0],job[1],job[2])) for job in morphJobs]
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# cv2.imwrite(str(endIndex) + 'Best Morph' + '.jpg',finalImg)

