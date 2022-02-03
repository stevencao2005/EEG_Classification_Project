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
from sklearn.model_selection import RandomizedSearchCV
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
from utils import calculate_metrics
from utils import case_by_case_analysis
from utils import create_directory
from utils import gettingInfoXGBoost


def fit_classifier():

    #LOAD AND PREPROCESS THE DATASET
    LoaderPreprocessor               = Load_And_Preprocess_Dataset()
    BED_dataset                      = LoaderPreprocessor.func_data_load()
    X_train, X_test, Y_train, Y_test = LoaderPreprocessor.func_dataPreProcessing(BED_dataset, toCategorical='false', subjectRemoval=sys.argv[2])
    Y_train, Y_test                  = np.squeeze(Y_train), np.squeeze(Y_test)

    #XGBOOST MODEL
    xgb_model  = xgb.XGBClassifier(objective="multi:softprob", learning_rate=0.036, random_state=42,
                                  n_estimators=400, max_depth=10, subsample=0.85)

    #TRAINING THE MODEL
    time1    = time.time()
    xgb_model.fit(X_train, Y_train)
    durationTrain = time.time() - time1
    print("time took to train the model", durationTrain)

    #EVALUATING THE MODEL
    time2   = time.time()
    y_pred  = xgb_model.predict(X_test)
    durationTest = time.time() - time2
    print("time took to go through the test dataset", durationTest)

    #GETTING THE PERFORMANCE OF THE MODEL
    metrics         = calculate_metrics(Y_test, y_pred, durationTrain, durationTest)
    confusionMatrix = confusion_matrix(Y_test, y_pred)
    print("confusion matrix\n", confusionMatrix)
    print("accuracy:  ", metrics['accuracy'][0])
    print("precision: ", metrics['precision'][0])
    print("recall:    ", metrics['recall'][0])
    print("f1_score:  ", metrics['f1 score'][0])

    # SAVING THE MODEL WEIGHTS AND INFO ABOUT THE TRIAL

    # create the directory
    file = os.path.abspath('.') + '/saved_datasets/XGBoost/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    create_directory(file)

    #saves the model weights
    file_name = file+'/model_weights_XGBoost.pkl'
    pickle.dump(xgb_model, open(file_name, "wb"))
    file_last = os.path.abspath('.') +'/saved_datasets/model_weights_XGBoost.pkl'
    pickle.dump(xgb_model, open(file_last, "wb"))

    #saves the parameters for this trial
    info = gettingInfoXGBoost(xgb_model)
    info.to_csv(file+'/info.csv', index=False)

    #saves the metrics for this trial
    metrics.to_csv(file+'/df_metrics.csv', index=False)


def fitted_classifier():

    #LOAD AND PREPROCESS THE DATASET
    LoaderPreprocessor               = Load_And_Preprocess_Dataset()
    BED_dataset                      = LoaderPreprocessor.func_data_load()
    X_train, X_test, Y_train, Y_test = LoaderPreprocessor.func_dataPreProcessing(BED_dataset, toCategorical='false', subjectRemoval=sys.argv[2])
    Y_train, Y_test                  = np.squeeze(Y_train), np.squeeze(Y_test)


    #LOAD THE ALREADY FITTED MODEL
    file_name        = os.path.abspath('.') + '/saved_datasets/model_weights_XGBoost.pkl'
    xgb_model_loaded = pickle.load(open(file_name, "rb"))

    #EVALUATING THE MODEL
    time2            = time.time()
    y_pred           = xgb_model_loaded.predict(X_test)
    durationTest     = time.time() - time2
    print("time took to go through the test dataset", durationTest)

    #GETTING THE PERFORMANCE OF THE MODEL
    metrics         = calculate_metrics(Y_test, y_pred, durationTest)
    print("accuracy:  ", metrics['accuracy'][0])
    print("precision: ", metrics['precision'][0])
    print("recall:    ", metrics['recall'][0])
    print("f1_score:  ", metrics['f1 score'][0])

    predictions     = case_by_case_analysis(Y_test, y_pred)

def tune_classifier():
    #LOAD AND PREPROCESS THE DATASET
    LoaderPreprocessor               = Load_And_Preprocess_Dataset()
    BED_dataset                      = LoaderPreprocessor.func_data_load_SPEC_RC_preprocessed()
    X_train, X_test, Y_train, Y_test = LoaderPreprocessor.func_dataPreProcessing(BED_dataset, toCategorical='false', subjectRemoval=sys.argv[2])
    Y_train, Y_test                  = np.squeeze(Y_train), np.squeeze(Y_test)


    #INITIALIZING THE XGBOOST MODEL
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", learning_rate=0.2, random_state=42,
                                  n_estimators=500, max_depth=10, subsample=0.6, seed=20)

    #INITIAZING THE PARAMETERS THAT THE XGBOOST MODEL IS GOING TO TRY
    params = {'max_depth': [10, 11],
              'learning_rate': [0.0380],
              'subsample': [0.75, 0.79, 0.81],
              'n_estimators': [400],
              'seed': [20],
              'random_state': [42]}

    #DECIDING HOW THE HYPER PARAMETERS WILL BE SELECTED, HOW IT BE EVALUATED, AND THE AMOUNT OF TIMES IT WILL TRY A COMBINATION
    clf = RandomizedSearchCV(estimator=xgb_model,
                             param_distributions=params,
                             scoring='neg_mean_squared_error',
                             n_iter=3,
                             verbose=1)

    #FITTING THE CLASSIFIER THE DIFFERENT COMBINATIONS OF HYPER PARAMETERS AND FIND WHICH ONE IS THE BEST
    clf.fit(X_train, Y_train)
    print("Best parameters:", clf.best_params_)
    print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))


if __name__ == '__main__':
    #STARTING CODE --------

    #DIRECTIONS ON WHAT TO DO AND WHETHER WE WANT TO USE THE DATA BEFORE OR AFTER SUBJECT REMOVAL
    #OPTIONS:
         #TRAIN OR LOAD
         #AFTER SUBJECT REMOVAL OR BEFORE SUBJECT REMOVAL
         #RO OR RC OR AS OR RC+RO OR AS+RC OR AS+RO OR AS+RO+RC
    sys.argv.extend(['train', 'after subject removal', 'RC'])

    if sys.argv[1] == 'train':
        #TRAINS A XGBOOST MODEL ON THE PREPROCESSED DATA
        fit_classifier()
    elif sys.argv[1] == 'load':
        #LOADS THE TRAINED XGBOOST MODEL
        fitted_classifier()
    elif sys.argv[1] == 'tune':
        #TUNE THE XGBOOST MODEL FOR THE BEST HYPER PARAMETERS
        tune_classifier()

    pass