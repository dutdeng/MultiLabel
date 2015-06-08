#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *
import random
import cPickle

def loadDataSet_CV(allFile,dim,labelNum):
	Record = []
	with open(allFile, 'r') as fr:
		for line in fr.readlines():
			record = zeros(dim+labelNum)
			lineList = line.strip().strip('{}').split(',')
			for i in range(len(lineList)):
				temp = lineList[i].split(' ')
				record[int(temp[0])] = int(temp[1])
			Record.append(record)
	Record = mat(Record)
	allIndex = [x for x in range(Record.shape[0])]
	trainIndex = random.sample([x for x in range(Record.shape[0])],Record.shape[0]/2)
	testIndex = list(set(allIndex).difference(set(trainIndex)))
	trainIndex.sort()
	print trainIndex
	print testIndex
	_trainDataSet = Record[trainIndex,:]
	_testDataSet = Record[testIndex,:]

	trainDataSet = (mat(_trainDataSet).T)[0:dim].T
	trainLabel = (mat(_trainDataSet).T)[dim:].T
	
	testDataSet = (mat(_testDataSet).T)[0:dim].T
	testLabel = (mat(_testDataSet).T)[dim:].T

	print trainDataSet
	print trainLabel
	print testDataSet
	print testLabel
	return mat(trainDataSet), mat(trainLabel), mat(testDataSet), mat(testLabel)

def initMatrix_H2(mat_V, mat_W):
	print mat_W.shape[0], mat_V.shape[0]
	mat_H = zeros((mat_W.shape[0], mat_V.shape[0]))
	count = 0
	LabelPro = zeros((mat_W.shape[0], mat_V.shape[0]))
	for i in range(mat_W.shape[0]):
		print i
		for j in range(mat_V.shape[0]):
			pro_Label = float(sum(mat_V[j]))/mat_V.shape[1]
			LabelPro[i,j] = pro_Label
			pro_Feature = float(sum(mat_W[i]))/mat_W.shape[1]
			#print AppFeatureMatrix[j], AppApiMatrix[i]
			if sum(mat_V[j]) != 0:
				pro_Feature_Label = float(sum(multiply(list(mat_V[j]),list(mat_W[i]))))/sum(mat_V[j]) #####
			else:
				pro_Feature_Label = 0
			if pro_Feature != 0:
				mat_H[i][j] = pro_Label * pro_Feature_Label / pro_Feature
			else:
				mat_H[i][j] = 0
	return mat(mat_H),mat(LabelPro)

def AppNmf(mat_V,mat_W,mat_H,maxIter=100):
	for i in range(maxIter):
		cost = sum(multiply(mat_V - mat_W * mat_H, mat_V - mat_W * mat_H))
		print i, ': ', cost
		mat_H = multiply(mat_H, (mat_W.T * mat_V) / (mat_W.T * mat_W * mat_H + 1e-16))		
	return mat_H

def AppNmfWeighting(mat_V,mat_W,mat_H,D,maxIter=100):
	for i in range(maxIter):
		costWeighting = multiply(mat_V - mat_W * mat_H, D)
		cost = sum(multiply(costWeighting, costWeighting))
		print i, ': ', cost
		mat_H = multiply(mat_H, (mat_W.T * multiply(mat_V, D)) / (mat_W.T * multiply(mat_W * mat_H, D) + 1e-16))		
	return mat_H

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = 0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = 0
    return retArray 

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels)
    D = mat(D)
    m,n = shape(dataMatrix)
    numSteps = 20.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    oldF1 = 0
    oldAcc = 0
    old_error1_0 = 0
    old_error0_1 = sum(labelMat)
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMax - float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr1 = multiply(abs(predictedVals - labelMat), D)
		errArr2 = multiply((predictedVals - labelMat), D)
		error1_0 = (sum(errArr1) + sum(errArr2)) /2
		error0_1 = (sum(errArr1) - sum(errArr2)) /2
		if sum(multiply(predictedVals, D)) != 0:
			precison = (sum(multiply(predictedVals, D)) - error1_0)/sum(multiply(predictedVals, D))
		else:
			precison = 1
		if sum(multiply(labelMat, D)) != 0:
			recall = (sum(multiply(labelMat, D)) - error0_1)/sum(multiply(labelMat, D))
		else:
			recall = 1
		weightedError = error1_0 + error0_1
		if (precison + recall) != 0:
			F1 = 2*precison*recall/(precison + recall)
		else:
			F1 = 0
		if (sum(multiply(labelMat, D)) + error1_0) != 0:
			acc = (sum(multiply(labelMat, D)) - error0_1) / (sum(multiply(labelMat, D)) + error1_0)
		else:
			acc = 1
                if weightedError < minError or (weightedError == minError and F1 > oldF1):
		    oldAcc = acc
		    oldF1 = F1
                    minError = weightedError
		    thresh = threshVal
		    print "---",minError,acc,F1,thresh
    return thresh#,minError#,bestClasEst


if __name__ == '__main__':
	trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet_CV('./enron.txt',1001,53)
	mat_H,LabelPro = initMatrix_H2(trainLabel.T,trainDataSet.T)

	D = ones((trainLabel.shape[0],trainLabel.shape[1]))/trainLabel.shape[0]
	print D
	
	DResult = zeros((testLabel.shape[0],testLabel.shape[1]))
	for aa in range(5):

		mat_H = AppNmfWeighting(trainLabel,trainDataSet,mat_H,D,50)

		trainResult = zeros((trainLabel.shape[0],trainLabel.shape[1]))
		thresh = zeros(trainLabel.shape[1])
		for i in range(trainLabel.shape[1]):
			trainResult = (trainDataSet*mat_H).T[i].T
			thresh[i] = buildStump(trainResult,(trainLabel.T[i].T),mat(D).T[i].T)
			print i,thresh[i]
		print thresh
	
		trainResult = zeros((trainLabel.shape[0],trainLabel.shape[1]))
		for j in range(trainResult.shape[1]):
			trainResult[nonzero((trainDataSet*mat_H)[:,j]>thresh[j])[0],j] = 1
		print sum(multiply(abs(trainResult - trainLabel),D),0)
		errorRatio = sum(multiply(abs(trainResult - trainLabel),D),0)
		alpha =  log((ones(errorRatio.shape[1]) - errorRatio)/maximum(errorRatio,tile(1e-16,(1,errorRatio.shape[1])))) * 0.5
		errorMatrix = 2*abs(trainResult - trainLabel) - 1
		print errorRatio
		print errorMatrix
		alphaMatrix = tile(alpha,(trainLabel.shape[0],1))
		print alphaMatrix
		expon = multiply(errorMatrix,alphaMatrix)
		D = multiply(D, exp(expon))
		D = D/sum(D,0)
		print D

		
		result = zeros((testLabel.shape[0],testLabel.shape[1]))
		result = testDataSet*mat_H

		_result = zeros((testLabel.shape[0],testLabel.shape[1])) - 1
		
		for i in range(result.shape[0]):
			_result[i,nonzero(result[i]>=result[i].max())[1]] = 1

		for j in range(result.shape[1]):
			_result[nonzero(result[:,j]>thresh[j])[0],j] = 1

		DResult += multiply(_result,alphaMatrix)


		raw_input()
	print DResult

	print trainLabel.shape[0],sum(trainLabel,0)

	testLabel = array(testLabel)
	print testLabel.shape[0],sum(testLabel,0)
	
	allRight = 0
	_sum = 0.0
	allPrecison = 0
	allRecall = 0
	allF1 = 0
	error11 = zeros(testLabel.shape[1])
	error22 = zeros(testLabel.shape[1])
	for i in range(result.shape[0]):
		error1 = 0
		error2 = 0
		#print result[i,]
		#print testLabel[i,]
		#print result[i,]>thresh
		#print sum(result[i,]>thresh)
		#raw_input()
		for j in range(result.shape[1]):
			if (DResult[i,j]>=0) and testLabel[i,j] == 0:
				error1 += 1
				error11[j] += 1
				#print result[i][j],testLabel[i][j]
			elif (DResult[i,j]<0) and testLabel[i,j] == 1:
				error2 += 1
				error22[j] += 1
				#print result[i][j],testLabel[i][j]
		_sum += float(sum(testLabel[i]) - error2)/(sum(testLabel[i]) + error1)
		if sum(testLabel[i]) != 0:
			recall = float(sum(testLabel[i]) - error2)/(sum(testLabel[i]))
		else:
			print i
			recall = 1
		allRecall += recall
		if (sum(testLabel[i]) + error1 - error2) != 0:
			precision = float(sum(testLabel[i]) - error2)/(sum(testLabel[i]) + error1 - error2)
			allPrecison += precision
		else:
			precision = 1
			allPrecison += 1
		if float(sum(testLabel[i]) - error2)/(sum(testLabel[i]) + error1) == 1:
			allRight += 1
		if (precision+recall) != 0:
			allF1 += 2*precision*recall/(precision+recall)
		else:
			allF1 += 0
	#print sum(testLabel),error1,error2
	#print float(error2)/int(sum(testLabel)), float(int(sum(testLabel)) - error2)/(int(sum(testLabel)) - error2 + error1), float(int(sum(testLabel)) - error2)/(int(sum(testLabel)) + error1)
	print sum(testLabel),testLabel.shape
	print "0-1 Loss:", 1-(float(allRight)/testLabel.shape[0])
	print "Hamming Loss:", (sum(error11) + sum(error22))/(testLabel.shape[0]*testLabel.shape[1])
	print "accuracy:", _sum/testLabel.shape[0]
	print "presion:", allPrecison/testLabel.shape[0]
	print "recall:", allRecall/testLabel.shape[0]
	print "F:", allF1/testLabel.shape[0]

	print error11,sum(error11)
	print error22,sum(error22)
	print sum(testLabel,0)
	aa = (sum(testLabel,0) - error22 + 1e-9) / (sum(testLabel,0) + 1e-9)
	bb = (sum(testLabel,0) - error22 + 1e-9) / (sum(testLabel,0) + error11 - error22 + 1e-9)
	print sum(aa)/testLabel.shape[1],sum(bb)/testLabel.shape[1]
	#print 2*multiply(aa,bb)/(aa+bb)
	print sum(2*multiply(aa,bb)/(aa+bb))/testLabel.shape[1]	
