import numpy as np
from prepro_dataConversion import *
import copy
def Augmentation(experimentList,labelList,labelListNormalised):
    #now let's work on the function for augmenting the data. I'm assuming the distribution derived by Michal
    #and Shrey are correct -> ie. 1st sensor at the top, then anti-clockwise distribution
    #Note, the coordinates are already normalized. Do a function at the very end that globalizes the coordinates
    for i in range(len(experimentList)):
        experimentList.append(copy.deepcopy(experimentList[i]))
        experimentList[-1].y = -1*experimentList[i].y
        experimentList[-1].s1 = experimentList[i].s3
        experimentList[-1].s3 = experimentList[i].s1

    #Initialize Augmented Matrices
    labelListAug1 = np.zeros((2,len(experimentList)))
    labelListNormalisedAug1 =np.zeros((2,len(experimentList)))

    #Apply the conditions after localizing
    labelList = localize(labelList)

    labelListAug1[0] = np.append(labelList[0],labelList[0,:])
    labelListAug1[1] = np.append(labelList[1],-1*labelList[1,:]) #flip the y axis

    labelListNormalisedAug1[0] = np.append(labelListNormalised[0],labelListNormalised[0,:])
    labelListNormalisedAug1[1] = np.append(labelListNormalised[1],-1*labelListNormalised[1,:]) #flip the y axis

    for i in range(len(experimentList)):
        experimentList.append(copy.deepcopy(experimentList[i]))
        experimentList[-1].x = -1*experimentList[i].x
        experimentList[-1].s4 = experimentList[i].s2
        experimentList[-1].s2 = experimentList[i].s4

    labelListAug2 = np.zeros((2,len(experimentList)))
    labelListNormalisedAug2 =np.zeros((2,len(experimentList)))

    labelListAug2[0] = np.append(labelListAug1[0],-1*labelListAug1[0,:]) #flip the x axis
    labelListAug2[1] = np.append(labelListAug1[1],labelListAug1[1,:])

    labelListNormalisedAug2[0] = np.append(labelListNormalisedAug1[0],-1*labelListNormalisedAug1[0,:]) #flip the x axis
    labelListNormalisedAug2[1] = np.append(labelListNormalisedAug1[1],labelListNormalisedAug1[1,:]) 
    return labelListAug2,labelListNormalisedAug2
