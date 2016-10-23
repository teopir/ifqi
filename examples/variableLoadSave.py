"""
This class will save and load the variables of an experiment
"""
import os
import os.path
import numpy as np
import json
import time

import cPickle

class ExperimentVariables:
    
    def __init__(self, experimentName): # type: (str) -> None
        """
        Given the experimentName this class organizes files to save on disk the results
        :param experimentName: name of the folder of the experiment
        """
        self.experimentName = experimentName
        self.regressorLoaded = []
        self.datasetLoaded = []
        self.iterationLoaded = []
        self.repetitionLoaded = []
        self.sizeLoaded = []
        self._loadDescriptor()
        
    def save(self, regressor, size, dataset, repetition, iteration, varName, value):
        """
        We save here the results of a variable
        :param regressor: number of the regressor in the json
        :param size: number of the size in the json
        :param dataset: number of the index
        :param repetition: number of the repetition
        :param iteration: number of the iteration
        :param varName: name of the variable to save
        :param value: value of the variable to save
        :return:
        """
        filename = self.experimentName + "/" + str(regressor) + str(varName) + "/" + str(size) + "/" + str(dataset) + "_" + str(repetition) + "_" + str(iteration) + ".npy"
        directory = os.path.dirname(filename)
        if not os.path.isdir(directory): os.makedirs(directory)
        np.save(filename, value)
        x=1.
        while True:
            try:
                self._loadDescriptor()
                if self._refreshDescriptor(size, iteration, repetition, dataset, regressor):
                    self._saveDescriptor()
                break
            except:
                pass
            x+=1.
            time.sleep(np.random.random()*x)

    def savePickle(self, regressor, size, dataset, repetition, iteration, varName, value):
        """
        We save here the results of a variable
        :param regressor: number of the regressor in the json
        :param size: number of the size in the json
        :param dataset: number of the index
        :param repetition: number of the repetition
        :param iteration: number of the iteration
        :param varName: name of the variable to save
        :param value: value of the variable to save
        :return:
        """
        filename = self.experimentName + "/" + str(regressor) + str(varName) + "/" + str(size) + "/" + str(dataset) + "_" + str(repetition) + "_" + str(iteration) + ".npy"
        directory = os.path.dirname(filename)
        if not os.path.isdir(directory): os.makedirs(directory)
        cPickle.dump(value,open(filename, "wb"))

    def loadPickle(self, regressor, size, dataset, repetition, iteration, varName):
        filename = self.experimentName + "/" + str(regressor) + str(varName) + "/" + str(size) + "/" + str(
            dataset) + "_" + str(repetition) + "_" + str(iteration) + ".npy"
        if os.path.isfile(filename):
            return cPickle.load(open(filename, "rb"))
        return False

    def loadSingle(self, regressor, size, dataset, repetition, iteration, varName):
        filename = self.experimentName + "/" + str(regressor) + str(varName) + "/" + str(size) + "/" + str(
            dataset) + "_" + str(repetition) + "_" + str(iteration) + ".npy"
        if os.path.isfile(filename):
            return np.load(filename)
        return False

    def load(self, regressor, size, iteration, varName):
        """
        Here you load already an "aggregate" value for the variable, computing the expected value over the repetitions and the datasets
        :param regressor: index of the regressor wished
        :param size: index of the size
        :param iteration: number of iteration
        :param varName: name of the variable
        :return: (mean, std, n, validity) where validity tells if n>0
        """
        self._loadDescriptor()
        values = []
        for dataset in self.datasetLoaded:
            for repetition in self.repetitionLoaded:
                    #TODO: replace with loadSingle
                    filename = self.experimentName + "/" + str(regressor) +  str(varName) + "/" + str(size) + "/" + str(dataset) + "_" + str(repetition) + "_" + str(iteration) + ".npy"
                    if os.path.isfile(filename):
                        values.append(np.load(filename))
        if len(values) > 0:
            mean = np.mean(values)
            std = np.std(values)
        
            return mean, std, len(values), True
        else:
            return 0., 0., 0, False

    def getSizeLines(self,varname):
        reg = {}
        for regressor in self.regressorLoaded:
            x = []
            y = []
            conf = []
            for size in self.sizeLoaded:
                iteration =  self.iterationLoaded[-1]
                mean, std, n, validity = self.load(regressor, size, iteration, varname)
                if validity:
                    y.append(mean)
                    x.append(size)
                    conf.append(std/np.sqrt(n)*1.96)
                if len(x) > 0:
                    reg[regressor] = (x,y,conf)
        return reg

    def getOverallSituation(self, varName):
        """
        This returns the situations of a variable in the experiment
        :param varName: name of the varialbe
        :return: { (a1, b1): (x1,y1,conf1), (a2,b2): (x2,y2,conf2), ... } basically a is the index of the regressor, size is the index of the size, x is a list with the iteration number and y a list with the values of the mean value of the variable for those iterations, conf the confidence of y
        """
        regSize =  {}
        for regressor in self.regressorLoaded:
            for size in self.sizeLoaded:
                x = []
                y = []
                conf = []
                for iteration in self.iterationLoaded:
                    mean, std, n, validity = self.load(regressor, size, iteration, varName)
                    if validity:
                        y.append(mean)
                        x.append(iteration)
                        conf.append(std/np.sqrt(n)*1.96)
                if len(x) > 0:
                    regSize[(regressor,size)] = (x,y,conf)
        return regSize
    
    def _loadDescriptor(self):
        filename = self.experimentName + "/descriptor.json"
        if os.path.isfile(filename):
            fp = open(filename, "r")
            data = json.load(fp)
            self.regressorLoaded = data["regressorLoaded"]
            self.repetitionLoaded = data["repetitionLoaded"]
            self.iterationLoaded = data["iterationLoaded"]
            self.datasetLoaded = data["datasetLoaded"]
            self.sizeLoaded = data["sizeLoaded"]
        else:
            self._saveDescriptor()
        
    def _saveDescriptor(self):
        data = {}
        data["regressorLoaded"] = self.regressorLoaded
        data["repetitionLoaded"] = self.repetitionLoaded
        data["iterationLoaded"] = self.iterationLoaded
        data["sizeLoaded"] = self.sizeLoaded
        data["datasetLoaded"] = self.datasetLoaded
        filename = self.experimentName + "/descriptor.json"
        directory = os.path.dirname(filename)
        if not os.path.isdir(directory): 
            os.makedirs(directory)
        fp = open(filename, "w")
        json.dump(data,fp)
    
    def _refreshDescriptor(self, size, iteration, repetition, dataset, regressor):
        ret = False
        if not regressor in self.regressorLoaded:
            ret = True
            self.regressorLoaded.append(regressor)
        if not repetition in self.repetitionLoaded:
            ret = True
            self.repetitionLoaded.append(repetition)
        if not iteration in self.iterationLoaded:
            ret = True
            self.iterationLoaded.append(iteration)
        if not dataset in self.datasetLoaded:
            ret = True
            self.datasetLoaded.append(dataset)
        if not size in self.sizeLoaded:
            ret = True
            self.sizeLoaded.append(size)
        return ret
            
    def aggregate(self, regressor):
        raise Exception("Not implemented yet")
    

        
