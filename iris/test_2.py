from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib.pyplot import *

#load iris data
iris = datasets.load_iris()
#check the dataset
print(iris.data.shape)
print(iris.target.shape)
print(iris.feature_names)





