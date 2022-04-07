from cgi import test
from copyreg import pickle
from multiprocessing import dummy
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
import pickle
import datetime


class TestPipeLine():
    def __init__(self, modelsBasePath, testFilePath):
        self.basePath = modelsBasePath
        self.modelFileList = glob.glob(self.basePath + "*.sav")
        self.testFilePath = testFilePath
        self.springModel = None
        self.fallModel = None
        self.summerModel = None
        self.winterModel = None
    
    def loadModels(self):
        print(self.modelFileList)
        for modelPath in self.modelFileList:
            print(modelPath)
            if "spring" in modelPath:
                springModelLoc = open(modelPath, 'rb')
                self.springModel = pickle.load(springModelLoc)
            if "summer" in modelPath:
                summerModelLoc = open(modelPath, 'rb')
                self.summerModel = pickle.load(summerModelLoc)
            if "fall" in modelPath:
                fallModelLoc = open(modelPath, 'rb')
                self.fallModel = pickle.load(fallModelLoc)
            if "winter" in modelPath:
                winterModelLoc = open(modelPath, 'rb')
                self.winterModel = pickle.load(winterModelLoc)
        
        
    def myDummyOHE(self):
        myTestDfX = pd.read_csv(self.testFilePath)
        myTestDfX["Ob"] = pd.to_datetime(myTestDfX["Ob"], infer_datetime_format=True)
        dummyDf = pd.get_dummies(myTestDfX.measure, prefix='measure')
        print("Shape of dummyDf: ", dummyDf.shape)
        print("dummyDf columns: ", dummyDf.columns)
        encodedData = pd.concat([myTestDfX[["Station", "Ob", "value", "R_flag", "I_flag", "Z_flag", 'B_flag']], dummyDf], axis = 1)
        # encodedData = encodedData.set_index("Ob")
        return encodedData

    def splitTestDf(self, myTestDfX):

        springStart = datetime.datetime(2021, 3, 1)
        springEnd = datetime.datetime(2021, 5, 31)
        summerStart = datetime.datetime(2021, 6, 1)
        summerEnd = datetime.datetime(2021, 8, 31)
        fallStart = datetime.datetime(2021, 9, 1)
        fallEnd = datetime.datetime(2021, 11, 30)
        winterStart = datetime.datetime(2021, 12, 1)
        winterEnd = datetime.datetime(2021, 12, 31)
        winter2Start = datetime.datetime(2021, 1, 1)
        winter2End = datetime.datetime(2021, 2, 28)

        springDf = myTestDfX[(myTestDfX["Ob"] >= springStart) & (myTestDfX["Ob"] <= springEnd)]
        summerDf = myTestDfX[(myTestDfX["Ob"] >= summerStart) & (myTestDfX["Ob"] <= summerEnd)]
        fallDf = myTestDfX[(myTestDfX["Ob"] >= fallStart) & (myTestDfX["Ob"] <= fallEnd)]
        winterDf = myTestDfX[(myTestDfX["Ob"] >= winterStart) & (myTestDfX["Ob"] <= winterEnd)]
        winter2Df = myTestDfX[(myTestDfX["Ob"] >= winter2Start) & (myTestDfX["Ob"] <= winter2End)]
        winterDf = pd.concat([winterDf, winter2Df])

        return springDf, summerDf, fallDf, winterDf
    

    def predictClasses(self, dataFrame):
        springDf, summerDf, fallDf, winterDf = self.splitTestDf(dataFrame)
        springPredict = self.springModel.predict(springDf)
        springPredictDf = pd.DataFrame(springPredict, index=springDf.index)

        summerPredict = self.summerModel.predict(summerDf)
        summerPredictDf = pd.DataFrame(summerPredict, index=summerDf.index)

        fallPredict = self.fallModel.predict(fallDf)
        fallPredictDf = pd.DataFrame(fallPredict, index=fallDf.index)

        winterPredict = self.winterModel.predict(winterDf)
        winterPredictDf = pd.DataFrame(winterPredict, index=winterDf.index)

        concatenatedDf = springPredictDf.concat([summerPredictDf, fallPredictDf, winterPredictDf], axis = 0)

        concatenatedDf.to_csv("predictedValues.csv")
        # print(springDf.head())
        # print(summerDf.head())
        # print(fallDf.head())
        # print(winterDf.head())





if __name__ == "__main__":
    testObj = TestPipeLine("/home/atharva/Atharva/NCSU/Sem2/ALDA/project/EcoNet1/Econet/models/", "/home/atharva/Atharva/NCSU/Sem2/ALDA/project/ECONet/test.csv")
    myDummyEncoded = testObj.myDummyOHE()
    testObj.loadModels()
    testObj.predictClasses(myDummyEncoded)

    """
    'measure_gust02', 'measure_gust10''measure_wd02', 'measure_wd10' these 4 features do not exist in the train data they only exist in the test data!!!
    """
