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

class trainPipeline():

    def __init__(self, myPath):
        self.myFileList = glob.glob(myPath)

    # def scaleData(self, dataFrame):
    #     myDf = dataFrame
    #     myScaler = StandardScaler()
    #     myDfX = dataFrame["target"]
    #     myDfY = dataFrame.drop(columns=["target"], axis = 1)
    #     myScaledData = myScaler.fit_transform(myDfX)
    #     return myScaledData
    
    def machineLearningModels(modelName):
        if modelName == "randomForest":
            randomForestModel = RandomForestClassifier(random_state=42)
            return randomForestModel
        
        elif modelName == "xgBoost":
            XGBoostModel = xgb.XGBClassifier()
            return XGBoostModel
    
    # def dummyEncoding(dataFrame):
    #     dummyDf = pd.get_dummies(dataFrame.measure, prefix='measure')
    #     encodedData = pd.concat([dataFrame[["Station", "Ob", "value", "target", "R_flag", "I_flag", "Z_flag", 'B_flag']], dummyDf], axis = 1)
    #     encodedData = encodedData.set_index("Ob")
    
    # def scaleDataFeature(onHotEncodedDf):         SCALE DATA WHEN MAKING THE CSVs
    #     for i in range(onHotEncodedDf[7:]):
    #         if onHotEncodedDf[i]

    
    def trainPipeLine(self):
        
        for i in range(self.myFileList):
            myFileName = os.path.basename(self.myFileList)
            readDf = pd.read_csv(self.myFileList[i])
            readX = readDf.drop(columns=["target"], axis = 1)
            readY = readDf["target"]
            XTrain, XVal, yTrain, yVal = train_test_split(readX, readY, stratify = readY, test_size=0.3)
            myModel1 = self.machineLearningModels("randomForest")
            myFit1 = myModel1.fit(XTrain, yTrain)
            myPredict1 = myModel1.predict(XVal)

            myAccuracy1 = accuracy_score(yVal, myPredict1)
            myF11 = f1_score(yVal, myPredict1)


            myModel2 = self.machineLearningModels("xgBoost")
            myFit2 = myModel2.fit(XTrain, yTrain)
            myPredict2 = myModel2.predict(XVal)

            myAccuracy2 = accuracy_score(yVal, myPredict2)
            myF12 = f1_score(yVal, myPredict2)

            if myAccuracy1 >= myAccuracy2 and myF11 >= myF12:
                pickle.dump(myModel1, open("/models/" + myFileName + ".sav"))
            
            elif myAccuracy1 <= myAccuracy2 and myF11 <= myF12:
                pickle.dump(myModel2, open("/models/" + myFileName + ".sav"))
            
            elif myF11 >= myF12:
                pickle.dump(myModel1, open("/models/" + myFileName + ".sav"))
            
            elif myF11 <= myF12:
                pickle.dump(myModel2, open("/models/" + myFileName + ".sav"))


            



            



    
    
    