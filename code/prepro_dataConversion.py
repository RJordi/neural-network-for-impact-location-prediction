import scipy.io
import os
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
#This function initializes the Experiment object and also converts the data from MATLAB to Python

class Experiment:
    def __init__(self,inputMat):
        self.time = inputMat[:,0]
        self.s1 = inputMat[:,1]
        self.s2 = inputMat[:,2]
        self.s3 = inputMat[:,3]
        self.s4 = inputMat[:,4]
        self.x = 0
        self.y = 0

def localize(labelList):
    labelListLoc = labelList - 250
    return labelListLoc

def globalize(labelList):
    labelListGlob = labelList + 250
    return labelListGlob

def dimensionalize(experimentList,h):
    for i in range(len(experimentList)):
        experimentList[i].x = h * experimentList[i].x
        experimentList[i].y = h * experimentList[i].y

def normalize(experimentList,h):
    for i in range(len(experimentList)):
        experimentList[i].x = experimentList[i].x/h
        experimentList[i].y = experimentList[i].y/h

def dataConversion(numDataDir):
    #numDataDir is the directory of the numerical data
    fileList = os.listdir(numDataDir) #Creates a list with the file names and directories

    #Now we only keep the .mat files
    for i in range(len(fileList)):
        if fileList[i][-4:] != ".mat":
            fileList.remove(fileList[i])

    labelList = np.zeros((2,len(fileList)))
    labelListNormalised = np.zeros((2,len(fileList)))
#The list is normalised in such a way that the coordinate is placed in the center of a 1x1 square
#Then in reality you would simply state the size of the domain that you are interested in and multiply it out and get realistic results-
#I think this makes sense - but discuss with the group/teacher

    for i in range(len(fileList)):
        #X coordinates
        labelList[0,i] = int(fileList[i][5:8])
        labelListNormalised[0,i] = (labelList[0,i] - 250)/150
        #Y coordinates
        labelList[1,i] = int(fileList[i][9:12])
        labelListNormalised[1,i] = (labelList[1,i] - 250)/150

    experimentList =[]
    for i in range(len(fileList)):
        matMatrix = scipy.io.loadmat(numDataDir + "\\" + fileList[i])
        experimentList.append(Experiment(matMatrix['num_data']))
        experimentList[i].x = labelListNormalised[0,i]
        experimentList[i].y = labelListNormalised[1,i]

    return experimentList,labelList,labelListNormalised



def ExpdataConversionVal(ExpDataDir,labListA):
    #ExoDataDir is the directory of the data
    fileList = os.listdir(ExpDataDir) #Creates a list with the file names and directories

    #Now we only keep the .mat files
    for i in range(len(fileList)):
        try:
            if fileList[i][-4:] != ".txt":
                fileList.remove(fileList[i])
        except:
            continue

    coordinateNo = len(fileList)
    lenData = 200
    sensorNo = 8
    experimentList = np.zeros((coordinateNo,sensorNo,lenData))
    labelList = np.zeros(coordinateNo)
    k=0
    for i in fileList:
        data_array = []
        j=0
        with open(ExpDataDir+i) as data:
            lines = data.readlines()[49980:49980+200] # min_val = 8 to delete text 49999:50199
            for line in lines:
                ls = line.split()
                experimentList[k][0][j] = float(ls[1].replace(',','.'))
                experimentList[k][1][j] = float(ls[2].replace(',','.'))
                experimentList[k][2][j] = float(ls[4].replace(',','.'))
                experimentList[k][3][j] = float(ls[3].replace(',','.'))
                j=j+1
        
        normalize = np.amax(np.abs([experimentList[k][0],experimentList[k][1],experimentList[k][2],experimentList[k][3]]))
        experimentList[k][0] = gaussian_filter(experimentList[k][0]/normalize,sigma = 3)
        experimentList[k][1] = gaussian_filter(experimentList[k][1]/normalize,sigma = 3)
        experimentList[k][2] = gaussian_filter(experimentList[k][2]/normalize,sigma = 3)
        experimentList[k][3] = gaussian_filter(experimentList[k][3]/normalize,sigma = 3)

               
        experimentList[k][4] =  np.gradient(experimentList[k][0])
        experimentList[k][5] =  np.gradient(experimentList[k][1])
        experimentList[k][6] =  np.gradient(experimentList[k][2])
        experimentList[k][7] =  np.gradient(experimentList[k][3])
        '''
        normalize = np.amax(np.abs([experimentList[k][4],experimentList[k][5],experimentList[k][6],experimentList[k][7]]))
        experimentList[k][4] = experimentList[k][4]/normalize
        experimentList[k][5] = experimentList[k][5]/normalize
        experimentList[k][6] = experimentList[k][6]/normalize
        experimentList[k][7] = experimentList[k][7]/normalize
        '''
        
        x = i[10:13]
        y = i[14:17]
        lab = kInverse(x,y,labListA)
        labelList[k] = lab
        k=k+1
    return labelList,experimentList



def kConvert(k,labListNA,GridSize,Translation):
    coord = np.zeros((2,))
    coordReal = np.zeros((2,))
    coord[0] = labListNA[0,k]
    coord[1] = labListNA[1,k]
    coordReal = coord*GridSize+Translation
    return coord,coordReal

def kInverse(x,y,labListA):
    test1 = np.where((labListA[0]+250) == int(x))
    test2 = np.where((labListA[1]+250) == int(y))
    print("x is supposedly" + str(x))
    print("y is supposedly" + str(y))
    for x in test1[0]:
        for y in test2[0]:
            if x == y:
                z = x
                break
    return z
