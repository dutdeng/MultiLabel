#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *
import random

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
	
def initMatrix_H2(mat_V, mat_W):
	print mat_W.shape[0], mat_V.shape[0]
	mat_H = zeros((mat_W.shape[0], mat_V.shape[0]))
	count = 0
	for i in range(mat_W.shape[0]):
		print i
		for j in range(mat_V.shape[0]):
			pro_Label = float(sum(mat_V[j]))/mat_V.shape[1]
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
	return mat(mat_H)

def AppNmf(mat_V,mat_W,mat_H,maxIter=100):
	for i in range(maxIter):
		cost = sum(multiply(mat_V - mat_W * mat_H, mat_V - mat_W * mat_H))
		print i, ': ', cost
		mat_H = multiply(mat_H, (mat_W.T * mat_V) / (mat_W.T * mat_W * mat_H + 1e-16))		
	return mat_H

	
if __name__ == '__main__':
	#trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet('./enron-train.txt','./enron-test.txt',1001,53)
	trainDataSet, trainLabel, testDataSet, testLabel = loadDataSet_CV('./delicious.txt',500,983)
	mat_H = initMatrix_H2(trainLabel.T,trainDataSet.T)
	mat_H = AppNmf(trainLabel,trainDataSet,mat_H,50)
	print sum(multiply(testDataSet*mat_H - testLabel, testDataSet*mat_H - testLabel))
	result = array(testDataSet*mat_H)
	testLabel = array(testLabel)
	error1 = 0
	error2 = 0
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			if result[i][j] >= 0.5 and testLabel[i][j] == 0:
				error1 += 1
				print result[i][j],testLabel[i][j]
			elif result[i][j] < 0.5 and testLabel[i][j] == 1:
				error2 += 1
				print result[i][j],testLabel[i][j]
	print sum(testLabel),error1,error2
	print float(error2)/int(sum(testLabel)), float(int(sum(testLabel)) - error2)/(int(sum(testLabel)) - error2 + error1)
