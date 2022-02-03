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


#Local application imports
from load_and_preprocess_data import Load_And_Preprocess_Dataset
from four_layer_neural_network import Net


from utils import case_by_case_analysis
from utils import create_directory
from utils import gettingInfoNN



def fit_classifier():

    #LOAD AND PREPROCESS THE DATASET
    LoaderPreprocessor               = Load_And_Preprocess_Dataset()
    BED_dataset                      = LoaderPreprocessor.func_data_load()
    X_train, X_test, Y_train, Y_test = LoaderPreprocessor.func_dataPreProcessing(BED_dataset, toCategorical='true', subjectRemoval=sys.argv[2])

    #IF PYTORCH, CREATE TENSOR DATASET
    train_loader, test_loader        = LoaderPreprocessor.func_createTensorDataset(X_train, X_test, Y_train, Y_test)

    #4-LAYERED NEURAL NETWORK USING PYTORCH
    model     = Net(in_features=X_train.shape[1], epochs=100)
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


    #SAVING THE MODEL WEIGHTS AND INFO ABOUT THE TRIAL

    #create the directory
    file = os.path.abspath('.') + '/saved_datasets/neural_network/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    create_directory(file)

    #saves the model weights
    file_name = file+'/model_weights_NN.pth'
    torch.save(model.state_dict(), file_name)
    file_last = os.path.abspath('.') +'/saved_datasets/model_weights_NN.pth'
    torch.save(model.state_dict(), file_last)

    #saves the parameters for this trial
    info = gettingInfoNN(model, optimizer)
    info.to_csv(file+'/info.csv', index=False)

    #saves the metrics for this trial
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
    file_name        = os.path.abspath('.') + '/saved_datasets/model_weights_NN.pth'
    model_loaded.load_state_dict(torch.load(file_name))


    #EVALUATING THE MODEL
    y_pred, y_true, metrics, confusionMatrix = model_loaded.compute_performance_metrics(test_loader)
    predictions                              = case_by_case_analysis(y_true, y_pred)

    print("accuracy:  ", metrics['accuracy'][0])
    print("precision: ", metrics['precision'][0])
    print("recall:    ", metrics['recall'][0])
    print("f1_score:  ", metrics['f1 score'][0])


if __name__ == '__main__':
    #STARTING CODE --------

    #DIRECTIONS ON WHAT TO DO AND WHETHER WE WANT TO USE THE DATA BEFORE OR AFTER SUBJECT REMOVAL
    #OPTIONS:
         #TRAIN OR LOAD
         #AFTER SUBJECT REMOVAL OR BEFORE SUBJECT REMOVAL
         #RO OR RC OR AS OR RC+RO OR AS+RC OR AS+RO OR AS+RO+RC
    sys.argv.extend(['train', 'after subject removal', 'RC'])

    if sys.argv[1] == 'train':
        #TRAINS A 4-LAYER NEURAL NETWORK ON THE PREPROCESSED DATA
        fit_classifier()

    elif sys.argv[1] == 'load':
        #LOADS THE TRAINED 4-LAYER NEURAL NETWORK
        fitted_classifier()

    pass

