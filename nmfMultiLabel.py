#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *
import random
import cPickle
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
		weightedError = error1_0 + error0_1
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
	trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet_num('./mediamill.txt',120,101)
	#trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet('./enron-train.txt','./enron-test.txt',1001,53)
	#trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet_CV2('./langLog.txt',1004,75)
	#trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet_CV('./tmc2007-500.txt',500,22)
	#mat_H,LabelPro = initMatrix_H2(trainLabel.T,trainDataSet.T)

	mat_H = ones((trainDataSet.shape[1], trainLabel.shape[1]))
	LabelPro = ones((trainDataSet.shape[1], trainLabel.shape[1]))
	newMat_H = (mat_H+1e-9)/(LabelPro+1e-9)
	print newMat_H
	_newMat_H = zeros((newMat_H.shape[0],newMat_H.shape[1]))
	_newMat_H[nonzero(newMat_H>1)[0],nonzero(newMat_H>1)[1]] = 1
	print _newMat_H

	#mat_H[nonzero(mat_H>=0.4)[0],nonzero(mat_H>=0.4)[1]] = 1
	#mat_H[nonzero(mat_H<0.25)[0],nonzero(mat_H<0.25)[1]] = 0
	#mat_H[nonzero(newMat_H>1)[0],nonzero(newMat_H>1)[1]] = 1
	#mat_H[nonzero(newMat_H<=1)[0],nonzero(newMat_H<=1)[1]] = 0
	print sum(mat_H)
	#print sum(mat_H,0)
	#print sum(mat_H,1)
	#print sum(_newMat_H,0)
	#print sum(_newMat_H,1)
	print nonzero((sum(mat_H,1))>3)
	print nonzero((sum(_newMat_H,1))>3)

	#mat_H = AppNmf(trainLabel,trainDataSet,mat_H,100)
	f = open('./trainDataSet.pkl','wb')
	cPickle.dump(trainDataSet,f,-1)
	f.close()
	f = open('./trainLabel.pkl','wb')
	cPickle.dump(trainLabel,f,-1)
	f.close()
	f = open('./testDataSet.pkl','wb')
	cPickle.dump(testDataSet,f,-1)
	f.close()
	f = open('./testLabel.pkl','wb')
	cPickle.dump(testLabel,f,-1)
	f.close()
	f = open('./mat_H.pkl','wb')
	cPickle.dump(mat_H,f,-1)
	f.close()
	f = open('./LabelPro.pkl','wb')
	cPickle.dump(LabelPro,f,-1)
	f.close()
	
	trainDataSet = cPickle.load(open('./trainDataSet.pkl', 'rb'))
	trainLabel = cPickle.load(open('./trainLabel.pkl', 'rb'))
	testDataSet = cPickle.load(open('./testDataSet.pkl', 'rb'))
	testLabel = cPickle.load(open('./testLabel.pkl', 'rb'))
	mat_H = cPickle.load(open('./mat_H.pkl', 'rb'))
	LabelPro = cPickle.load(open('./LabelPro.pkl', 'rb'))
	# 1
	mat_H = AppNmf(trainLabel,trainDataSet,mat_H,50)
	print mat_H
	#print sum(mat_H,0)
	#print sum(mat_H,1)
	new_= zeros((mat_H.shape[0],mat_H.shape[1]))
	new_[nonzero(mat_H>0)[0],nonzero(mat_H>0)[1]] = 1
	print sum(new_,0)
	
	result = zeros((testLabel.shape[0],testLabel.shape[1]))	
	result = testDataSet*mat_H
	
	trainResult = zeros((trainLabel.shape[0],trainLabel.shape[1]))
	thresh = zeros(trainLabel.shape[1])
	for i in range(trainLabel.shape[1]):
		trainResult = (trainDataSet*mat_H).T[i].T
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

	print error11,sum(error11)
	print error22,sum(error22)
	print sum(testLabel,0)
	aa = (sum(testLabel,0) - error22 + 1e-9) / (sum(testLabel,0) + 1e-9)
	bb = (sum(testLabel,0) - error22 + 1e-9) / (sum(testLabel,0) + error11 - error22 + 1e-9)
	print sum(aa)/testLabel.shape[1],sum(bb)/testLabel.shape[1]
	#print 2*multiply(aa,bb)/(aa+bb)
	print sum(2*multiply(aa,bb)/(aa+bb))/testLabel.shape[1]
	
	'''
	error1 = 0
	error2 = 0
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			if result[i][j] >= 0.8 and testLabel[i][j] == 0:
				error1 += 1
				print result[i][j],testLabel[i][j]
			elif result[i][j] < 0.8 and testLabel[i][j] == 1:
				error2 += 1
				print result[i][j],testLabel[i][j]
	print error1,error2,(sum(testLabel)-error2)/(sum(testLabel) +error1)
	'''
