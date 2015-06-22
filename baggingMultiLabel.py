#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *
import random
import cPickle
import randMat
def loadDataSet(trainFile,testFile,dim,labelNum):
	Record = []
	with open(trainFile, 'r') as fr:
		for line in fr.readlines():
			record = zeros(dim+labelNum)
			lineList = line.strip().strip('{}').split(',')
			for i in range(len(lineList)):
				temp = lineList[i].split(' ')
				record[int(temp[0])] = int(temp[1])
			Record.append(record)
	
	trainDataSet = (mat(Record).T)[0:dim].T
	trainLabel = (mat(Record).T)[dim:].T

	Record = []
	with open(testFile, 'r') as fr:
		for line in fr.readlines():
			record = zeros(dim+labelNum)
			lineList = line.strip().strip('{}').split(',')
			for i in range(len(lineList)):
				temp = lineList[i].split(' ')
				record[int(temp[0])] = int(temp[1])
			Record.append(record)

	testDataSet = (mat(Record).T)[0:dim].T
	testLabel = (mat(Record).T)[dim:].T

	print trainDataSet
	print trainLabel
	print testDataSet
	print testLabel
	return mat(trainDataSet), mat(trainLabel), mat(testDataSet), mat(testLabel)

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

def Bagging(trainDataSet,trainLabel):
	sample = zeros((trainDataSet.shape[0],trainDataSet.shape[1]))
	label = zeros((trainLabel.shape[0],trainLabel.shape[1]))
	for i in range(trainDataSet.shape[0]):
		index = random.randint(0,trainDataSet.shape[0]-1)
		sample[i] = trainDataSet[index]
		label[i] = trainLabel[index]
	return mat(sample), mat(label)


def loadDataSet_CV2(allFile,dim,labelNum):
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

	trainLabel = (mat(_trainDataSet).T)[0:labelNum].T
	trainDataSet = (mat(_trainDataSet).T)[labelNum:].T
	
	testLabel = (mat(_testDataSet).T)[0:labelNum].T
	testDataSet = (mat(_testDataSet).T)[labelNum:].T

	print trainDataSet
	print trainLabel
	print testDataSet
	print testLabel
	return mat(trainDataSet), mat(trainLabel), mat(testDataSet), mat(testLabel)

def loadDataSet_num(allFile,dim,labelNum):
	Record = []
	with open(allFile, 'r') as fr:
		for line in fr.readlines():
			record = zeros(dim+labelNum)
			lineList = line.strip().split(',')
			for i in range(len(lineList)):
				record[i] = lineList[i]
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

def AppNmf2(mat_V,mat_W,mat_H,maxIter=100):
	mat_VV = array(mat_V)
	index = list(nonzero(mat_VV<1)[0])
	zero_index = random.sample([x for x in list(nonzero(mat_VV<1)[0])],len(index)/20)
	mat_VV[zero_index] = 1
	for i in range(maxIter):
		mat_V_ = multiply(mat_W * mat_H, mat_VV)
		cost = sum(multiply(mat_V - mat_V_, mat_V - mat_V_))
		print i, ': ', cost
		mat_H = multiply(mat_H, (mat_W.T * mat_V) / (mat_W.T * mat_V_ + 1e-16))		
	return mat_H

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = 0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = 0
    return retArray
    

def buildStump(dataArr,classLabels):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels)
    m,n = shape(dataMatrix)
    #print m,n
    numSteps = 20.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    oldF1 = 0
    oldAcc = 0
    old_error1_0 = 0
    old_error0_1 = sum(labelMat)
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']: 
                threshVal = (rangeMax - float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr1 = abs(predictedVals - labelMat)
		errArr2 = (predictedVals - labelMat)
		error1_0 = (sum(errArr1) + sum(errArr2)) /2
		error0_1 = (sum(errArr1) - sum(errArr2)) /2
		if sum(predictedVals) != 0:
			precison = (sum(predictedVals) - error1_0)/sum(predictedVals)
		else:
			precison = 1
		if sum(labelMat) != 0:
			recall = (sum(labelMat) - error0_1)/sum(labelMat)
		else:
			recall = 1
		weightedError = 0.5*error1_0 + error0_1
		if (precison + recall) != 0:
			F1 = 2*precison*recall/(precison + recall)
		else:
			F1 = 0
		if (sum(labelMat) + error1_0) != 0:
			acc = (sum(labelMat) - error0_1) / (sum(labelMat) + error1_0)
		else:
			acc = 1
                #if weightedError < minError or (weightedError == minError and F1 > oldF1):
		#if F1 > oldF1:
		if acc>oldAcc:
		    oldAcc = acc
		    oldF1 = F1
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
		    thresh = threshVal
		    print "---",minError,acc,F1,thresh
		    #print sum(errArr1),sum(errArr2),sum(labelMat)
    return thresh#,minError#,bestClasEst


if __name__ == '__main__':
	#trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet_num('./scene.txt',294,6)
	#trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet('./enron-train.txt','./enron-test.txt',1001,53)
	#trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet_CV2('./langLog.txt',1004,75)
	trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet_CV('./enron.txt',1001,53)
	result = zeros((testLabel.shape[0],testLabel.shape[1]))	
	for it in range(10):
	
		_trainDataSet,_trainLabel = Bagging(trainDataSet,trainLabel)
		#mat_H = ones((_trainDataSet.shape[1], _trainLabel.shape[1]))
		print _trainDataSet.shape[1], _trainLabel.shape[1]
		mat_H = randMat.randMat(_trainDataSet.shape[1], _trainLabel.shape[1])
		mat_H = AppNmf(_trainLabel,_trainDataSet,mat_H,50)
	
		trainResult = zeros((_trainLabel.shape[0],_trainLabel.shape[1]))
		thresh = zeros(_trainLabel.shape[1])
		for i in range(_trainLabel.shape[1]):
			trainResult = (_trainDataSet*mat_H).T[i].T
			thresh[i] = buildStump(trainResult,(_trainLabel.T[i].T))
			print i,thresh[i]
		print thresh

		trainResult = zeros((testLabel.shape[0],testLabel.shape[1]))

		_result = zeros((testLabel.shape[0],testLabel.shape[1]))
		_result = testDataSet*mat_H

		for i in range(_result.shape[0]):
			trainResult[i,nonzero(_result[i]>=_result[i].max())[1]] = 1

		for j in range(trainResult.shape[1]):
			trainResult[nonzero((testDataSet*mat_H)[:,j]>thresh[j])[0],j] = 1

		print sum(abs(trainResult-testLabel)),sum(trainResult-testLabel)

		result += trainResult


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
		for i in range(trainResult.shape[0]):
			error1 = 0
			error2 = 0
			for j in range(trainResult.shape[1]):
				if (trainResult[i,j]==1) and testLabel[i,j] == 0:
					error1 += 1
					error11[j] += 1
					#print result[i][j],testLabel[i][j]
				elif (trainResult[i,j]==0) and testLabel[i,j] == 1:
					error2 += 1
					error22[j] += 1
					#print result[i][j],testLabel[i][j]
			if (sum(testLabel[i]) + error1) == 0:
				_sum += 1
			else:
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
			if float(sum(testLabel[i]) - error2)/(sum(testLabel[i]) + error1) == 1 or (sum(testLabel[i]) + error1) == 0:
				allRight += 1
			if (precision+recall) != 0:
				allF1 += 2*precision*recall/(precision+recall)
			else:
				allF1 += 0
		print sum(testLabel),testLabel.shape
		print "0-1 Loss:", 1-(float(allRight)/testLabel.shape[0])
		print "Hamming Loss:", (sum(error11) + sum(error22))/(testLabel.shape[0]*testLabel.shape[1])
		print "accuracy:", _sum/testLabel.shape[0]
		print "presion:", allPrecison/testLabel.shape[0]
		print "recall:", allRecall/testLabel.shape[0]
		print "F:", allF1/testLabel.shape[0]


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
		for j in range(result.shape[1]):
			if (result[i,j]>=5) and testLabel[i,j] == 0:
				error1 += 1
				error11[j] += 1
				#print result[i][j],testLabel[i][j]
			elif (result[i,j]<5) and testLabel[i,j] == 1:
				error2 += 1
				error22[j] += 1
				#print result[i][j],testLabel[i][j]
		if (sum(testLabel[i]) + error1) == 0:
			_sum += 1
		else:
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
		#if float(sum(testLabel[i]) - error2)/(sum(testLabel[i]) + error1) == 1 or (sum(testLabel[i]) + error1) == 0:
		if error1 == 0 and error2 == 0:
			allRight += 1
		if (precision+recall) != 0:
			allF1 += 2*precision*recall/(precision+recall)
		else:
			allF1 += 0
	print sum(testLabel),testLabel.shape
	print "0-1 Loss:", 1-(float(allRight)/testLabel.shape[0])
	print "Hamming Loss:", (sum(error11) + sum(error22))/(testLabel.shape[0]*testLabel.shape[1])
	print "accuracy:", _sum/testLabel.shape[0]
	print "presion:", allPrecison/testLabel.shape[0]
	print "recall:", allRecall/testLabel.shape[0]
	print "F:", allF1/testLabel.shape[0]

	#print error11,sum(error11)
	#print error22,sum(error22)
	print sum(testLabel,0)
	aa = (sum(testLabel,0) - error22 + 1e-9) / (sum(testLabel,0) + 1e-9)
	bb = (sum(testLabel,0) - error22 + 1e-9) / (sum(testLabel,0) + error11 - error22 + 1e-9)
	print sum(aa)/testLabel.shape[1],sum(bb)/testLabel.shape[1]

	print sum(2*multiply(aa,bb)/(aa+bb))/testLabel.shape[1]

