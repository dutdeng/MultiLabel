#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *
import random
import cPickle

def loadDataSet():
	trainDataSet = []
	trainLable = []
	with open('./scene_train', 'r') as fr:
		for line in fr.readlines():
			record = zeros(294)
			recordLable = zeros(6)
			lineList = line.strip().split(' ')
			lableList = lineList[0].split(',')
			for i in range(len(lableList)):
				#print int(lableList[i])
				recordLable[int(lableList[i])] = 1
			for i in range(len(lineList)-1):
				temp = lineList[i+1].split(':')
				if float(temp[1]) < 0.3:
					record[int(temp[0]) - 1]  = 0
				elif float(temp[1]) > 0.7:
					record[int(temp[0]) - 1]  = 1
				else:
					record[int(temp[0]) - 1] = 0.5
			trainDataSet.append(record)
			trainLable.append(recordLable)
	print mat(trainDataSet)
	print mat(trainLable)
	
	testDataSet = []
	testLable = []
	with open('./scene_test', 'r') as fr:
		for line in fr.readlines():
			record = zeros(294)
			recordLable = zeros(6)
			lineList = line.strip().split(' ')
			lableList = lineList[0].split(',')
			for i in range(len(lableList)):
				#print int(lableList[i])
				recordLable[int(lableList[i])] = 1
			for i in range(len(lineList)-1):
				temp = lineList[i+1].split(':')
				record[int(temp[0]) - 1] = float(temp[1])
			testDataSet.append(record)
			testLable.append(recordLable)
	print mat(testDataSet)
	print mat(testLable)

	return mat(trainDataSet), mat(trainLable), mat(testDataSet), mat(testLable)

def loadDataSet2():
	trainDataSet = []
	trainLable = []
	with open('./train.txt', 'r') as fr:
		for line in fr.readlines():
			record = zeros(1185,dtype=int32)
			recordLable = zeros(27,dtype=int32)
			lineList = line.strip().split(',')
			for i in range(len(lineList)):
				if lineList[i] == 'YES':
					record[i-1] = 1
				elif lineList[i] == 'NO':
					record[i-1] = 0
				elif lineList[i] == '1':
					recordLable[i-1186] = 1
				elif lineList[i] == '0':
					recordLable[i-1186] = 0
			trainDataSet.append(record)
			trainLable.append(recordLable)
	print mat(trainDataSet).shape
	print mat(trainLable)

	testDataSet = []
	testLable = []
	with open('./test.txt', 'r') as fr:
		for line in fr.readlines():
			record = zeros(1185,dtype=int32)
			recordLable = zeros(27,dtype=int32)
			lineList = line.strip().split(',')
			for i in range(len(lineList)):
				if lineList[i] == 'YES':
					record[i-1] = 1
				elif lineList[i] == 'NO':
					record[i-1] = 0
				elif lineList[i] == '1':
					recordLable[i-1186] = 1
				elif lineList[i] == '0':
					recordLable[i-1186] = 0
			testDataSet.append(record)
			testLable.append(recordLable)
	print mat(testDataSet).shape
	print mat(testLable)

	return mat(trainDataSet), mat(trainLable), mat(testDataSet), mat(testLable)

def loadDataSet3():
	Record = []
	RecordLable = []
	with open('./train.txt', 'r') as fr:
		for line in fr.readlines():
			record = zeros(1185,dtype=int32)
			recordLable = zeros(27,dtype=int32)
			lineList = line.strip().split(',')
			for i in range(len(lineList)):
				if lineList[i] == 'YES':
					record[i-1] = 1
				elif lineList[i] == 'NO':
					record[i-1] = 0
				elif lineList[i] == '1':
					recordLable[i-1186] = 1
				elif lineList[i] == '0':
					recordLable[i-1186] = 0
			Record.append(record)
			RecordLable.append(recordLable)
	with open('./test.txt', 'r') as fr:
		for line in fr.readlines():
			record = zeros(1185,dtype=int32)
			recordLable = zeros(27,dtype=int32)
			lineList = line.strip().split(',')
			for i in range(len(lineList)):
				if lineList[i] == 'YES':
					record[i-1] = 1
				elif lineList[i] == 'NO':
					record[i-1] = 0
				elif lineList[i] == '1':
					recordLable[i-1186] = 1
				elif lineList[i] == '0':
					recordLable[i-1186] = 0
			Record.append(record)
			RecordLable.append(recordLable)
	Record = mat(Record)
	RecordLable = mat(RecordLable)
	allIndex = [x for x in range(Record.shape[0])]
	trainIndex = random.sample([x for x in range(Record.shape[0])],Record.shape[0]/2)
	testIndex = list(set(allIndex).difference(set(trainIndex)))
	trainIndex.sort()
	print trainIndex
	print testIndex
	trainDataSet = Record[trainIndex,:]
	testDataSet = Record[testIndex,:]

	trainLabel = RecordLable[trainIndex,:]
	testLabel = RecordLable[testIndex,:]	

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
	trainIndex = random.sample([x for x in range(Record.shape[0])],60*Record.shape[0]/100)
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
	trainIndex = random.sample([x for x in range(Record.shape[0])],60*Record.shape[0]/100)
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

def AppNmf(mat_V,mat_W,mat_H,maxIter=5):
	for i in range(maxIter):
		cost = sum(multiply(mat_V - mat_W * mat_H, mat_V - mat_W * mat_H))
		#print i, ': ', cost
		mat_H = multiply(mat_H, (mat_W.T * mat_V) / (mat_W.T * mat_W * mat_H + 1e-16))		
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
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
	#print rangeMin,rangeMax
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMax - float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
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
                #weightedError = sum(errArr1)  #calc total error multiplied by D
		weightedError = 0.75*error1_0 + error0_1
		if (precison + recall) != 0:
			F1 = 2*precison*recall/(precison + recall)
		else:
			F1 = 0
		if (sum(labelMat) + error1_0) != 0:
			acc = (sum(labelMat) - error0_1) / (sum(labelMat) + error1_0)
		else:
			acc = 1
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError or (weightedError == minError and F1 > oldF1):
		#if F1 > oldF1:
		#if acc>oldAcc:
		    oldAcc = acc
		    oldF1 = F1
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                   # bestStump['dim'] = i
                   # bestStump['thresh'] = threshVal
                   # bestStump['ineq'] = inequal
		    thresh = threshVal
		    print "---",minError,acc,F1,thresh
		    #print sum(errArr1),sum(errArr2),sum(labelMat)
    return thresh#,minError#,bestClasEst

if __name__ == '__main__':
	#trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet3()
	trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet_CV('./bibtex.txt',1836,159)
	#trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet_CV2('./langLog.txt',1004,75)
	#trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet_num('./mediamill.txt',120,101)
	mat_H1 = ones((trainDataSet.shape[1], trainLabel.shape[1]))/120
	mat_H2 = ones((trainDataSet.shape[1], trainLabel.shape[1]))/(120*2)
	mat_W1 = trainDataSet
	mat_W2 = ones((trainDataSet.shape[0],trainDataSet.shape[1])) - trainDataSet
	for i in range(10):
		mat_V1 = trainLabel - mat_W1*mat_H1
		mat_V1 = maximum(mat_V1,zeros((mat_V1.shape[0], mat_V1.shape[1])))
		mat_H2 = AppNmf(mat_V1,mat_W2,mat_H2,1)
		cost = sum(multiply(trainLabel - mat_W1 * mat_H1 - mat_W2 * mat_H2, trainLabel - mat_W1 * mat_H1 - mat_W2 * mat_H2))
		print cost
		mat_V2 = trainLabel - mat_W2*mat_H2
		mat_V2 = maximum(mat_V2,zeros((mat_V2.shape[0], mat_V2.shape[1])))
		mat_H1 = AppNmf(mat_V2,mat_W1,mat_H1,5)
		cost = sum(multiply(trainLabel - mat_W1 * mat_H1 - mat_W2 * mat_H2, trainLabel - mat_W1 * mat_H1 - mat_W2 * mat_H2))
		print cost


	result = testDataSet*mat_H1 + (ones((testDataSet.shape[0],testDataSet.shape[1]))-testDataSet)*mat_H2
	
	trainResult = zeros((trainLabel.shape[0],trainLabel.shape[1]))
	thresh = zeros(trainLabel.shape[1])
	for i in range(trainLabel.shape[1]):
		trainResult = (trainDataSet*mat_H1 + (ones((trainDataSet.shape[0],trainDataSet.shape[1]))-trainDataSet)*mat_H2).T[i].T
		#for j in range(trainResult.shape[0]):
		#	print trainResult[j],(trainLabel.T[i].T)[j]
		#raw_input()
		thresh[i] = buildStump(trainResult,(trainLabel.T[i].T))
		print i,thresh[i]
	print thresh
	# 1
	'''
	trainResult = zeros((trainLabel.shape[0],trainLabel.shape[1]))
	for j in range(trainResult.shape[1]):
		trainResult[nonzero((trainDataSet*mat_H)[:,j]>thresh[j])[0],nonzero((trainDataSet*mat_H)[:,j]>thresh[j])[1]] = 1

	errorRatio = sum(abs(trainResult - trainLabel),0)/trainLabel.shape[0]
	'''
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	for i in range(result.shape[0]):
		result[i,nonzero(result[i]>=result[i].max())[1]] = 1

	#for i in range(trainResult.shape[0]):
	#	print trainResult[i],(trainLabel.T[0].T)[i]
	#result[nonzero((testDataSet*mat_H)>=0.3)[0],nonzero((testDataSet*mat_H)>=0.3)[1]] = 1
	#result[:,labelIndex] = 0
	
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
			if (result[i,j]==1 or result[i,j]>thresh[j]) and testLabel[i,j] == 0:
				error1 += 1
				error11[j] += 1
				#print result[i][j],testLabel[i][j]
			elif (result[i,j]!=1 and result[i,j]<=thresh[j]) and testLabel[i,j] == 1:
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
	#print sum(testLabel),error1,error2
	#print float(error2)/int(sum(testLabel)), float(int(sum(testLabel)) - error2)/(int(sum(testLabel)) - error2 + error1), float(int(sum(testLabel)) - error2)/(int(sum(testLabel)) + error1)
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
	#print 2*multiply(aa,bb)/(aa+bb)
	print sum(2*multiply(aa,bb)/(aa+bb))/testLabel.shape[1]
	
	

