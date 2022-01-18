
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
from sklearn.model_selection import RandomizedSearchCV

from load_and_preprocess_data import Load_And_Preprocess_Dataset

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
    subjectNames             = [int(i) for i in labels]
    subjectNames_str         = [str(int(i)) for i in labels]

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

def gettingInfo(xgb_model):
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

    #CREATE THE DIRECTORY
    file = os.path.abspath('.') + '/saved_datasets/XGBoost/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    create_directory(file)
    #SAVES THE MODEL WEIGHTS
    file_name = file+'/RC_model_weights_XGBoost.pkl'
    pickle.dump(xgb_model, open(file_name, "wb"))
    file_last = os.path.abspath('.') +'/saved_datasets/RC_model_weights_XGBoost.pkl'
    pickle.dump(xgb_model, open(file_last, "wb"))

    #SAVES THE PARAMETERS FOR THIS TRIAL
    info = gettingInfo(xgb_model)
    info.to_csv(file+'/info.csv', index=False)

    #SAVES THE METRICS FOR THIS TRIAL
    metrics.to_csv(file+'/df_metrics.csv', index=False)
    i=1

def fitted_classifier():

    #LOAD AND PREPROCESS THE DATASET
    LoaderPreprocessor               = Load_And_Preprocess_Dataset()
    BED_dataset                      = LoaderPreprocessor.func_data_load()
    X_train, X_test, Y_train, Y_test = LoaderPreprocessor.func_dataPreProcessing(BED_dataset, toCategorical='false', subjectRemoval=sys.argv[2])
    Y_train, Y_test                  = np.squeeze(Y_train), np.squeeze(Y_test)


    #LOAD THE ALREADY FITTED MODEL
    file_name        = os.path.abspath('.') + '/saved_datasets/RC_model_weights_XGBoost.pkl'
    xgb_model_loaded = pickle.load(open(file_name, "rb"))

    #EVALUATING THE MODEL
    time2        = time.time()
    y_pred       = xgb_model_loaded.predict(X_test)
    durationTest = time.time() - time2
    print("time took to go through the test dataset", durationTest)

    #GETTING THE PERFORMANCE OF THE MODEL
    metrics         = calculate_metrics(Y_test, y_pred, durationTest)
    print("accuracy:  ", metrics['accuracy'][0])
    print("precision: ", metrics['precision'][0])
    print("recall:    ", metrics['recall'][0])
    print("f1_score:  ", metrics['f1 score'][0])

    predictions = case_by_case_analysis(Y_test, y_pred)

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
    sys.argv.extend(['train', 'before subject removal', 'RC'])

    if sys.argv[1] == 'train':
        #TRAINS A XGBOOST MODEL ON THE PREPROCESSED DATA
        fit_classifier()
    elif sys.argv[1] == 'load':
        #LOADS THE TRAINED XGBOOST MODEL
        fitted_classifier()
    elif sys.argv[1] == 'tune':
        #TUNE THE XGBOOST MODEL FOR THE BEST HYPER PARAMETERS
        tune_classifier()
    i =1

    pass