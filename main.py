
# coding: utf-8

# In[27]:

from barycentric import generateMorphBarycentric
from measureDifference import getOptimalSequence
import cv2
import numpy as np

endIndex = 15
seq = getOptimalSequence(endIndex)
finalImg = cv2.imread('./dataSet/test_' + str(seq[0])+'.png')	# append very first img


# In[28]:

print(seq)
for i in range(len(seq)):
	if i != len(seq)-1:
		halfway = generateMorphBarycentric(seq[i],seq[i+1])
		finalImg = np.concatenate((finalImg,halfway),axis=1)
		rhsImg = cv2.imread('./dataSet/test_' + str(seq[i+1])+'.png')
		finalImg = np.concatenate((finalImg,rhsImg),axis=1)

cv2.imwrite(str(endIndex) + 'Best Morph' + '.jpg',finalImg)

