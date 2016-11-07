from numpy import *
import matplotlib.pyplot as plt
import os
def loadDataSet(fileName, delim=' '):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    labelMat=[]
    datArr=[]
    for line1 in stringArr[1:]:
        labelMat.append(line1[0])
        data=[map(float,x) for x in line1[1:]]
        datArr.append(data)
    return mat(datArr),labelMat

def pca(dataMat, topNfeat):
    meanVals = mean(dataMat,0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved,rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

if __name__=='__main__':
    path = os.path.abspath('./')+'/file/meide.bin'
    dataArr,label=loadDataSet(path)
    result,result1= pca(dataArr, 4)
    print result,result1
