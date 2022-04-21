from base64 import encode
from copyreg import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import TomekLinks 
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate


class splitScaleData():
    def __init__(self, myPath) -> None:
        self.myPath = myPath 

    def readTrain(self):
        fullTrainDf = pd.read_csv(self.myPath)
        return fullTrainDf
    
    """
    1: Onehot encode measures
    2: Scale data depending on measures
    3: Filter by seasons (using datetime library??)
    4: Save the split season file each by season name (Dec-Feb, Mar-May, Jun-Aug, Sep-Nov) in seasonsDataFolder
    5: Use each season dataframe filter by 43 stations
    6: save all 43*4 CSVs in splitDataFolder 
    """

    def myDummyOHE(self, dataFrame):
        dummyDf = pd.get_dummies(dataFrame.measure, prefix='measure')
        encodedData = pd.concat([dataFrame[["Station", "Ob", "value", "target", "R_flag", "I_flag", "Z_flag", 'B_flag']], dummyDf], axis = 1)
        # encodedData = encodedData.set_index("Ob")
        return encodedData
    
    def scaleDataFeature(self, oneHotEncodedDf):         #SCALE DATA WHEN MAKING THE CSVs
        for columnName in oneHotEncodedDf.columns[-16:-1]:
            # if columnName != "measure_blackglobetemp":
                # continue
            # print(i, oneHotEncodedDf[i])
            # columnName = oneHotEncodedDf[i]
            print("Column Name LIst: ", columnName)
            test = oneHotEncodedDf.loc[oneHotEncodedDf[columnName] == 1]
            # print("Test Data Head:",test.head())
            print("Test Data Head Shape:",test.shape)
            # break
    
    
            # if onHotEncodedDf[i]
    
    def indvScaledData(self, oneHotEncodedDf):
        encodedDataX = oneHotEncodedDf.drop(columns=["target"], axis=1)
        encodedDataX1 = encodedDataX.drop(columns=["Ob", "Station"], axis=1)
        encodedDataY = oneHotEncodedDf["target"]
        for columns in encodedDataX1.columns[-16:-1]:
            existsDf = encodedDataX1.loc[encodedDataX1[columns] == 1]
            print(columns)
            # print(existsDf)
            myScaler = StandardScaler()
            myFit = pd.DataFrame(myScaler.fit_transform(existsDf), columns=encodedDataX1.columns, index = existsDf.index)
            # print(myFit)
            encodedDataX["values"] = encodedDataX.merge(myFit["values"], "right", encodedDataX.index)
        
        return encodedDataX
        

myObj = splitScaleData('train.csv')

myReadDf = myObj.readTrain()

myDummyONE = myObj.myDummyOHE(myReadDf)

# test1 = myDummyONE.loc[myDummyONE["measure_blackglobetemp"] == 1]
# print(test1)

# myScaledDf = myObj.scaleDataFeature(myDummyONE)
myScaledDataX = myObj.indvScaledData(myDummyONE)

print(myScaledDataX.head())