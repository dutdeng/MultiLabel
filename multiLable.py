#! /usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *

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

def initMatrix_H(mat_V, mat_W):
	print mat_W.shape[0], mat_V.shape[0]
	mat_H = zeros((mat_W.shape[0], mat_V.shape[0]))
	count = 0
	for i in range(mat_W.shape[0]):
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

def initMatrix_H2(mat_V, mat_W):
	print mat_W.shape[0], mat_V.shape[0]
	mat_H = zeros((mat_W.shape[0], mat_V.shape[0]))
	count = 0
	for i in range(mat_W.shape[0]):
		for j in range(mat_V.shape[0]):
			pro_Label = float(sum(mat_V[j]))/mat_V.shape[1]
			pro_Feature = float(sum(mat_W[i]))/mat_W.shape[1]
			#print mat_V[j],mat_W[i]
			if sum(mat_V[j]) != 0:
				pro_Feature_Label = float(sum(bitwise_and(list(mat_V[j]),list(mat_W[i]))))/sum(mat_V[j]) #####
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

def AppNmf2(mat_V,mat_W,mat_H,maxIter=100):
	for i in range(maxIter):
		mat_v_ = multiply(mat_W * mat_H, mat_V)
		cost = sum(multiply(mat_V - mat_v_, mat_V - mat_v_))
		print i, ': ', cost
		mat_H = multiply(mat_H, (mat_W.T * mat_V) / (mat_W.T * mat_v_ + 1e-16))		
	return mat_H

if __name__ == '__main__':
	'''
	trainDataSet, trainLable, testDataSet, testLable = loadDataSet()
	mat_H = initMatrix_H(trainLable.T,trainDataSet.T)
	mat_H = AppNmf(trainLable,trainDataSet,mat_H)
	print trainLable
	print sum(trainLable)
	print trainDataSet*mat_H 
	'''
	trainDataSet, trainLable, testDataSet, testLable = loadDataSet2()
	mat_H = initMatrix_H2(trainLable.T,trainDataSet.T)
	mat_H = AppNmf(trainLable,trainDataSet,mat_H,50)
	print sum(multiply(testDataSet*mat_H - testLable, testDataSet*mat_H - testLable))
	result = array(testDataSet*mat_H)
	testLable = array(testLable)
	error1 = 0
	error2 = 0
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			if result[i][j] >= 0.5 and testLable[i][j] == 0:
				error1 += 1
				print result[i][j],testLable[i][j]
			elif result[i][j] < 0.5 and testLable[i][j] == 1:
				error2 += 1
				print result[i][j],testLable[i][j]
	print sum(testLable),error1,error2
