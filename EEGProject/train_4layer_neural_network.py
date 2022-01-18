
#IMPORT ALL NEEDED MODULES
import scipy
import time
import mne
import torch
import sys
import os
import pathlib
import datetime
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import librosa
from librosa.feature.spectral import spectral_centroid
from librosa.feature.spectral import spectral_flatness
from librosa.feature.spectral import spectral_bandwidth
from librosa.feature.spectral import rms


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tensorflow.keras.utils import to_categorical
from torch.utils.data import TensorDataset
from scipy.io import loadmat
from pyprep.prep_pipeline import PrepPipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from load_and_preprocess_data import Load_And_Preprocess_Dataset
from four_layer_neural_network import Net

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
def gettingInfo(model, optimizer):
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

def fit_classifier():

    #LOAD AND PREPROCESS THE DATASET
    LoaderPreprocessor               = Load_And_Preprocess_Dataset()
    BED_dataset                      = LoaderPreprocessor.func_data_load()
    X_train, X_test, Y_train, Y_test = LoaderPreprocessor.func_dataPreProcessing(BED_dataset, toCategorical='true', subjectRemoval=sys.argv[2])

    #IF PYTORCH, CREATE TENSOR DATASET
    train_loader, test_loader        = LoaderPreprocessor.func_createTensorDataset(X_train, X_test, Y_train, Y_test)

    #4-LAYERED NEURAL NETWORK USING PYTORCH
    model     = Net(in_features=X_train.shape[1], epochs=1900)
    optimizer = optim.SGD(model.parameters(), lr=0.055, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()


    #TRAINING THE MODEL
    time1    = time.time()
    model.train(train_loader, optimizer, criterion)
    duration = time.time() - time1
    print("time took to train the model", duration)

    #EVALUATING THE MODEL
    output_list, true_list, metrics, confusionMatrix = model.compute_performance_metrics(test_loader)
    print("confusion matrix\n", confusionMatrix)
    print("accuracy:  ", metrics['accuracy'][0])
    print("precision: ", metrics['precision'][0])
    print("recall:    ", metrics['recall'][0])
    print("f1_score:  ", metrics['f1 score'][0])

    #CREATE THE DIRECTORY
    file = os.path.abspath('.') + '/saved_datasets/neural_network/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    create_directory(file)
    #SAVES THE MODEL WEIGHTS
    file_name = file+'/RC_model_weights_NN.pth'
    torch.save(model.state_dict(), file_name)

    #SAVES THE PARAMETERS FOR THIS TRIAL
    info = gettingInfo(model, optimizer)
    info.to_csv(file+'/info.csv', index=False)

    #SAVES THE METRICS FOR THIS TRIAL
    metrics.to_csv(file+'/df_metrics.csv', index=False)

def fitted_classifier():

    #LOAD AND PREPROCESS THE DATASET
    LoaderPreprocessor               = Load_And_Preprocess_Dataset()
    BED_dataset                      = LoaderPreprocessor.func_data_load()
    X_train, X_test, Y_train, Y_test = LoaderPreprocessor.func_dataPreProcessing(BED_dataset, toCategorical='true', subjectRemoval=sys.argv[2])

    #IF PYTORCH, CREATE TENSOR DATASET
    train_loader, test_loader        = LoaderPreprocessor.func_createTensorDataset(X_train, X_test, Y_train, Y_test)

    #4-LAYERED NEURAL NETWORK USING PYTORCH
    model_loaded     = Net(in_features=X_train.shape[1], epochs=1900)
    optimizer        = optim.SGD(model_loaded.parameters(), lr=0.055, weight_decay=0.0001)
    criterion        = nn.CrossEntropyLoss()

    #LOAD THE ALREADY FITTED MODEL
    file_name        = os.path.abspath('.') + '/saved_datasets/RC_model_weights_NN.pth'
    model_loaded.load_state_dict(torch.load(file_name))


    #EVALUATING THE MODEL
    y_pred, y_true, metrics, confusionMatrix = model_loaded.compute_performance_metrics(test_loader)
    predictions                              = case_by_case_analysis(y_true, y_pred)

    print("accuracy:  ", metrics['accuracy'][0])
    print("precision: ", metrics['precision'][0])
    print("recall:    ", metrics['recall'][0])
    print("f1_score:  ", metrics['f1 score'][0])



    i=1

def main():
    pass

if __name__ == '__main__':
    #STARTING CODE --------

    #DIRECTIONS ON WHAT TO DO AND WHETHER WE WANT TO USE THE DATA BEFORE OR AFTER SUBJECT REMOVAL
    #OPTIONS:
         #TRAIN OR LOAD
         #AFTER SUBJECT REMOVAL OR BEFORE SUBJECT REMOVAL
         #RO OR RC OR AS OR RC+RO OR AS+RC OR AS+RO
    sys.argv.extend(['train', 'after subject removal', 'AS+RO'])

    if sys.argv[1] == 'train':
        #TRAINS A 4-LAYER NEURAL NETWORK ON THE PREPROCESSED DATA
        fit_classifier()

    elif sys.argv[1] == 'load':
        #LOADS THE TRAINED 4-LAYER NEURAL NETWORK
        fitted_classifier()
    i =1

    pass

