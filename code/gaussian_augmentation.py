import numpy as np
from scipy.ndimage.filters import gaussian_filter

def gaussian_augmentation(experimentList,labelList,labelListNormalised):

	new_labelList = np.zeros((labelList.shape[0],labelList.shape[1]*2))
	new_labelListNormalised = np.zeros((labelListNormalised.shape[0],labelListNormalised.shape[1]*2))

	for i in range (len(experimentList)):
		'''
		experimentList[i].s1 = gaussian_filter(experimentList[i].s1, sigma=2)
		experimentList[i].s2 = gaussian_filter(experimentList[i].s2, sigma=2)
		experimentList[i].s3 = gaussian_filter(experimentList[i].s3, sigma=2)
		experimentList[i].s4 = gaussian_filter(experimentList[i].s4, sigma=2)
		'''
		#print(len(experimentList[i].s1))
		noise = np.random.normal(0,0.02,20000)
		experimentList[i].s1 += noise
		experimentList[i].s2 += noise
		experimentList[i].s3 += noise
		experimentList[i].s4 += noise

		experimentList.append(experimentList[i])


	new_labelList[0] = np.hstack((labelList[0], labelList[0]))
	new_labelList[1] = np.hstack((labelList[1], labelList[1]))

	new_labelListNormalised[0] = np.hstack((labelListNormalised[0], labelListNormalised[0]))
	new_labelListNormalised[1] = np.hstack((labelListNormalised[1], labelListNormalised[1]))

	return experimentList, new_labelList, new_labelListNormalised