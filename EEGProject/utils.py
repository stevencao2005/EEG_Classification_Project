"""
@author: Steven Cao"""


#IMPORT ALL NEEDED MODULES
#Standard library imports
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import sys
import scipy
import time
import warnings

matplotlib.use("TkAgg")
warnings.filterwarnings('ignore')

#Third party imports
import librosa
from librosa.feature.spectral import spectral_centroid
from librosa.feature.spectral import spectral_flatness
from librosa.feature.spectral import spectral_bandwidth
from librosa.feature.spectral import rms
import mne
from pyprep.prep_pipeline import PrepPipeline
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import xgboost as xgb


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path

def gettingInfoNN(model, optimizer):
    info = pd.DataFrame(data=np.zeros((1, 8), dtype=np.float), index=[0],
                       columns=['lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'epochs', 'subject removal', 'data'])
    info['lr']              = optimizer.defaults['lr']
    info['momentum']        = optimizer.defaults['momentum']
    info['dampening']       = optimizer.defaults['dampening']
    info['weight_decay']    = optimizer.defaults['weight_decay']
    info['nesterov']        = optimizer.defaults['nesterov']
    info['epochs']          = model.epochs
    info['subject removal'] = sys.argv[2]
    info['data']            = sys.argv[3]
    i=1
    return info

def gettingInfoXGBoost(xgb_model):
    info = pd.DataFrame(data=np.zeros((1, 8), dtype=np.float), index=[0],
                       columns=['objective', 'learning_rate', 'random_state', 'n_estimators', 'max_depth',
                                'subsample', 'subject removal', 'data'])
    info['objective']          = xgb_model.objective
    info['learning_rate']      = xgb_model.learning_rate
    info['random_state']       = xgb_model.random_state
    info['n_estimators']       = xgb_model.n_estimators
    info['max_depth']          = xgb_model.max_depth
    info['subsample']          = xgb_model.subsample
    info['subject removal']    = sys.argv[2]
    info['data']               = sys.argv[3]
    i=1
    return info

def calculate_metrics(y_true, y_pred, durationTrain=None, durationTest=None):
    metrics = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                           columns=['accuracy', 'precision', 'recall', 'f1 score', 'training_time', 'testing_time'])

    labels = list(np.unique(y_true))
    metrics['accuracy']      = accuracy_score(y_true, y_pred)
    metrics['recall']        = recall_score(y_true, y_pred, labels=labels, average='macro')
    metrics['precision']     = precision_score(y_true, y_pred, labels=labels, average='macro')
    metrics['f1 score']      = f1_score(y_true, y_pred, labels=labels, average='macro')
    metrics['training_time'] = durationTrain
    metrics['testing_time']  = durationTest

    return metrics

def case_by_case_analysis(y_true, y_pred):
    correctPredictionIndexes = []
    labels                   = np.unique(y_true)
    indexes                  = np.arange(1, len(y_true) + 1)
    subjectNames             = [i for i in labels]
    subjectNames_str         = [str(i) for i in labels]

    #GETTING THE CONFUSION MATRIX
    confusionMX              = confusion_matrix(y_true, y_pred)
    print("confusion matrix\n", confusionMX)

    #GETTING THE CLASSIFICATION REPORT
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=subjectNames_str))

    #GETTING THE AMOUNT OF SAMPLES PREDICTED CORRECTLY FOR EACH SUBJECT
    print("amount of samples predicted correctly for each subject")
    for i in range(confusionMX.shape[0]):
        print('    subject' + str(subjectNames[i]), confusionMX[i, i])

    #GETTING THE PREDICTIONS FOR EACH SAMPLE AND GROUP THEM ON WHETHER IT WAS CORRECT OR INCORRECT
    predictions = dict()
    for name in subjectNames:
        predictions[name] = dict()
        predictions[name]['correct predictions']   = {}
        predictions[name]['incorrect predictions'] = {}
    for i, value in enumerate(y_true):
        for j in subjectNames:
            if value == j:
                subjectName = value
        if y_pred[i] == y_true[i]:
            predictions[subjectName]['correct predictions'][i]   = (indexes[i] - 1, y_pred[i], y_true[i])
        else:
            predictions[subjectName]['incorrect predictions'][i] = (indexes[i] - 1, y_pred[i], y_true[i])

    #GETTING THE AMOUNT OF TIMES THE MODEL PREDICTED FOR EACH SUBJECT FOR EACH SUBJECTS'S SAMPLES
    subjects        = subjectNames.copy()
    subjectClassify = subjectNames.copy()
    values          = dict()
    for subjectName in subjects:
        values[subjectName] = dict()
        for subject in subjectClassify:
            values[subjectName][subject] = 0
    for subjectName in subjectNames:
        subject = predictions[subjectName]['incorrect predictions']
        for sample in list(subject.items()):
            predictedValue = sample[1][1]
            for i in subjects:
                if predictedValue == i:
                    values[subjectName][i] = values[subjectName][i] + 1

    #PRINTING THE RESULTS FROM THE ABOVE CODE OUT
    for subjectName in subjectNames:
        subject = values[subjectName]
        print('subject', subjectName)
        for subjectName in subjectClassify:
            print('   {0}   {1}'.format(subjectName, subject[subjectName]))

    #GETTING THE COHEN KAPPA SCORE
    from sklearn.metrics import cohen_kappa_score
    Coh_k_s = cohen_kappa_score(y_true, y_pred)

    #GETTING THE MATTHEWS CORRELATION COEFFICIENT
    from sklearn.metrics import matthews_corrcoef
    Mat_corf = matthews_corrcoef(y_true, y_pred)

    #IF DISPLAY IS 1, THEN WILL PRINT EACH SAMPLE, THE PREDICTED SUBJECT, AND THE ACTUAL SUBJECT
    display = 0
    if display == 1:
        for i in predictions:
            for j in predictions[i]:
                for k in predictions[i][j]:
                    sampleNum = predictions[i][j][k]
                    print("Sample {0}: Predicted as {1}  Actual Value is {2}".format(sampleNum[0], sampleNum[1],
                                                                                     sampleNum[2]))
    return predictions
