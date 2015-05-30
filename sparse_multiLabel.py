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
				record[int(temp[0]) - 1] = float(temp[1])
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
	with open('./b.txt','r') as fr:
		for line in fr.readlines():
			lineList = line.strip().split('\t')
			record = zeros(100)
			for i in range(len(lineList)):
				record[i] = float(lineList[i])
			trainDataSet.append(list(record))
	print mat(trainDataSet)
	return mat(trainDataSet)

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


def AppNmf(mat_V,mat_W,mat_H,maxIter=100):
	for i in range(maxIter):
		cost = sum(multiply(mat_V - mat_W * mat_H, mat_V - mat_W * mat_H))
		print i, ': ', cost
		mat_H = multiply(mat_H, (mat_W.T * mat_V) / (mat_W.T * mat_W * mat_H + 1e-16))		
	return mat_H

def AppNmf2(mat_V,mat_W,mat_H,maxIter=100):
	for i in range(maxIter):
		cost = sum(multiply(mat_V - mat_W * mat_H, mat_V - mat_W * mat_H))
		print i, ': ', cost	
		mat_W = multiply(mat_W, (mat_V * mat_H.T) / (mat_W * mat_H * mat_H.T + 1e-16))	
	return mat_W

def projfunc(s, k1, k2):
	s = array(s)
	N = len(s)
	v = s + (k1-sum(s))/N
	print "s:",s
	#print k1
	#print sum(s)
	#print "N:",N
	#print "v:",v
	zerocoeff = []
	j = 0
	while 1:
		midpoint = ones((N,1))*k1/(N - len(zerocoeff))
		midpoint[zerocoeff] = 0
		w = v - midpoint
		print '1',w
		#print '2',v
		a = sum(power(w,2))
		temp_v = v.reshape((1,v.shape[0]))
		b = 2*dot(temp_v,w)
		c = sum(power(v,2))-k2
		if a == 0:
			alpha = 0
			break
		else:
			alpha = (-b+real(sqrt(b**2-4*a*c)))/(2*a)
		#print alpha
		v = alpha*w + v
		if all(v>=0):
			usedIters = j+1
			print "!"
			print v
			break
		j += 1
		#print j
		#print v
		#print "zerocoeff",zerocoeff
		#print nonzero(v<=0)[0]
		zerocoeff.extend(nonzero(v<=0)[0])
		#print zerocoeff
		#print v.shape
		v[zerocoeff] = 0
		tempsum = sum(v)
		v = v + (k1-tempsum)/(N-len(zerocoeff))
    		v[zerocoeff] = 0
	return v

def randMat(line,r):
	mat_ = random.random((line,r))
	return mat(mat_)
'''
def nmfsc(V, rdim, sW, sH, maxIter=100):
	samples = V.shape[0]
	vdim = V.shape[1]
	
	W = randMat(samples,rdim)
	H = randMat(rdim,vdim)

	if sW != 0:
		L1a = sqrt(samples)-(sqrt(samples)-1)*sW
		for i in range(rdim):
			#print W[:,i],L1a
			W[:,i] = projfunc(W[:,i],L1a,1)
	if sH != 0:
		L1s = sqrt(vdim)-(sqrt(vdim)-1)*sH
		for i in range(rdim):
			print H[i,:],L1s
			H[i,:] = (projfunc(H[i,:].T,L1s,1)).T
	
	print V - W*H
	stepsizeW = 1
	stepsizeH = 1
	cost = sum(multiply(V - W*H, V - W*H))
	
	for i in range(maxIter):
		cost = sum(multiply(V - W*H, V - W*H))
		print i, ': ', cost
		#raw_input()
		if sH != 0:
			dH = W.T*(W*H - V)
			while 1:
				newH = H - stepsizeH*dH
				for i in range(rdim):
					newH[i,:] = (projfunc(newH[i,:].T,L1s,1)).T
				newCost = sum(multiply(V - W*newH, V - W*newH))
				if newCost <= cost:
					break;
				stepsizeH = stepsizeH/2
			stepsizeH = stepsizeH*1.2
			H = newH
		else:
			H = multiply(H, (W.T*V)/(W.T*W*H+1e-9))
		norms = sqrt(sum(power(H.T,2),0))
		print "norms:", norms
		H = H/(norms.T*ones((1,vdim)));
		W = multiply(W,(ones((samples,1))*norms))	

		cost = sum(multiply(V - W*H, V - W*H))
		print i, ': ', cost
		if sW != 0:
			dW = (W*H - V)*H.T
			while 1:
				newW = W - stepsizeW*dW
				norms = sqrt(sum(power(newW,2),0))
				print "norms:", norms
				print norms.shape
				for i in range(rdim):
					newW[:,i] = projfunc(newW[:,i],L1a*norms[0,i],norms[0,i]**2)
				newCost = sum(multiply(V - newW*H, V - newW*H))
				if newCost <= cost:
					break;
				stepsizeW = stepsizeW/2
			stepsizeW = stepsizeW*1.2
			W = newW
		else:
			W = multiply(W, (V*H.T)/(W*H*H.T+1e-9))
		
'''				
if __name__ == '__main__':
	trainDataSet, trainLable, testDataSet, testLable = loadDataSet()
	#nmfsc(trainDataSet,40,0.5,0.5)
	'''
	with open('./trainData.txt','w') as fw:
		for i in range(trainDataSet.shape[0]):
			for j in range(trainDataSet.shape[1]):
				fw.write(str(trainDataSet[i,j])+' ')
			fw.write('\n')
	
	trainDataSet = loadDataSet2()
	mat_H = initMatrix_H(trainLable.T,trainDataSet.T)
	print mat_H
	mat_H = AppNmf(trainLable,trainDataSet,mat_H,100)
	print trainDataSet*mat_H
	'''
	mat_H = randMat(trainLable.shape[1],trainDataSet.shape[1])
	mat_H = AppNmf(trainDataSet, trainLable, mat_H)
	print sum(multiply(testDataSet, testDataSet))
	#mat_W = randMat(testDataSet.shape[0],mat_H.shape[0])
	#mat_W = AppNmf2(testDataSet,mat_W, mat_H)
	print testLable
	#print testDataSet[0]
	newTestLabel = []
	for i in range(testDataSet.shape[0]):
		Label = [0,0,0,0,0,0]
		min = sum(multiply(testDataSet[i],testDataSet[i]))
		min_index = -1
		for j in range(trainLable.shape[1]):
			Label = [0,0,0,0,0,0]
			Label[j] = 1
			cost = sum(multiply(Label*mat_H - testDataSet[i], Label*mat_H - testDataSet[i]))
			if cost < min:
				min = cost
				min_index = j
		Label = [0,0,0,0,0,0]
		Label[min_index] = 1
		newTestLabel.append(Label)
	print mat(newTestLabel)
	print sum(testLable)
	print sum(mat(newTestLabel))
	print sum(abs(mat(newTestLabel) - testLable))
	print sum(mat(newTestLabel) - testLable)
