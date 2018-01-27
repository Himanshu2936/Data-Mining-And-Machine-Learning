import numpy as np
import pandas as pd
import csv

data=[]

def main():
	read_data()
	X,Y=train_data()
	theta=solve_by_normal_equation(X,Y)
	error=error_function_value(theta)
	sample=[0.085,13.0, 10.5, 1.0, 0.8, 4.78, 39.0, 5.5, 5.5, 331.0, 13.3,390.5,17.71]
	answer=get_value_for_sample(sample,theta)
	print("Values of theta from theta0 to theta13 are : ")
	print(theta)
	print()
	print("Value of Cost Function = ",error)
	print()
	print("Answer for smaple case ",end=' ')
	print(sample,end=' ')
	print(" is : ",answer)

def read_data():
	with open('boston_housing.csv', 'rt') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		next(reader, None)
		for row in reader:
			temp = [float(i) for i in row]
			data.append(temp)

def train_data():
	trainingData = pd.DataFrame(data, columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','y'])
	trainingData.insert(0, 'x0', np.ones(506))
	X = trainingData[['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13']]
	Y = trainingData[['y']]
	return X,Y
	
def solve_by_normal_equation(X,Y):
	XtranX = X.T.dot(X)
	XtranXinv= np.linalg.inv(XtranX)
	XtranXinvXtran = XtranXinv.dot(X.T)
	theta =XtranXinvXtran.dot(Y)
	return theta

def error_function_value(theta):
	sum=0
	for row in data:
		s=theta[0]
		for i in range(1,14,1):
			s=s+theta[i]*row[i-1]
		s=s-row[13]
		sum=sum+s*s
	return (sum/(2*506))
	
def get_value_for_sample(sample,theta):
	s=theta[0]
	for i in range(1,14,1):
		s=s+theta[i]*sample[i-1]
	return s
if __name__ == "__main__":
    main()