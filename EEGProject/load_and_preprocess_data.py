
#IMPORT ALL NEEDED MODULES
import time
import scipy
import math
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from scipy.io import loadmat
from sklearn.utils import shuffle



class Load_And_Preprocess_Dataset():

    def __init__(self):
        pass

    def func_data_load(self):
        if sys.argv[3] == 'RC':
            BED_dataset = self.func_data_load_SPEC_RC_preprocessed()
            return BED_dataset
        elif sys.argv[3] == 'RO':
            BED_dataset = self.func_data_load_SPEC_RO_preprocessed()
            return BED_dataset
        elif sys.argv[3] == 'RC+RO':
            BED_dataset = self.func_data_load_SPEC_RC_and_RO_preprocessed()
            return BED_dataset


    def func_data_load_SPEC_RC_preprocessed(self):
        BED_dataset = loadmat(os.path.abspath('.') + '/BED/Features/Identification/SPEC/SPEC_rest_closed.mat')
        return BED_dataset

    def func_data_load_SPEC_RO_preprocessed(self):
        BED_dataset = loadmat(os.path.abspath('.') + '/BED/Features/Identification/SPEC/SPEC_rest_open.mat')
        return BED_dataset

    def func_data_load_SPEC_RC_and_RO_preprocessed(self):
        RC_dataset = loadmat(os.path.abspath('.')+'/BED/Features/Identification/SPEC/SPEC_rest_closed.mat')
        RO_dataset = loadmat(os.path.abspath('.')+'/BED/Features/Identification/SPEC/SPEC_rest_open.mat')

        RC_features = RC_dataset['feat']
        RC_labels   = RC_dataset['Y']
        RC_info     = self.func_data_getinfo(RC_dataset)

        RO_features = RO_dataset['feat']
        RO_labels   = RO_dataset['Y']
        RO_info     = self.func_data_getinfo(RO_dataset)


        result      = [min(els) for els in zip(RC_info[1], RO_info[1])]

        RO_and_RC_dataset_features = np.zeros((RC_features.shape[0], 448))
        RO_and_RC_dataset_labels   = np.zeros((RC_features.shape[0], 2))

        i=1
        for j in range(len(RC_info[1]) + 1):
            threshold = sum(RC_info[1][:j])
            if i <= threshold:
                startInd = sum(RC_info[1][:j])
                endInd   = sum(RC_info[1][:j + 1])
                if endInd == 2875:
                    endInd = 2876
                m = startInd
                for k in range(int(endInd - startInd)):  # k=1
                    rcMinimums         = sum(RC_info[1][:j])
                    roMinimums         = sum(RO_info[1][:j])
                    oneSampleRC        = RC_features[rcMinimums + k, :]
                    oneSampleRO        = RO_features[roMinimums + k, :]
                    oneSampleRO_and_RC = np.append(oneSampleRC, oneSampleRO)

                    n                                = int(math.floor(j / 3) + 1)
                    sessions                         = [1, 2, 3]
                    nn                               = int(sessions[j % 3])
                    oneSampleRO_and_RC_label         = (n, nn)
                    RO_and_RC_dataset_features[i, :] = oneSampleRO_and_RC
                    RO_and_RC_dataset_labels[i, :]   = oneSampleRO_and_RC_label
                    i += 1

        RO_and_RC_dataset = {'feat': RO_and_RC_dataset_features, 'Y': RO_and_RC_dataset_labels}

        return RO_and_RC_dataset

    def func_data_getinfo(self, BED_dataset):
        info = dict()
        all_SPECdataset_names = list(BED_dataset.keys())
        info['all_SPECdataset_names'] = all_SPECdataset_names
        info["all_parts_of_SPEC_dataset"] = all_SPECdataset_names[3:]
        info['INFO'] = dict()
        info['INFO']['shapeOfAllSamples'] = BED_dataset['INFO'].shape
        info['INFO']['shapeOfOneSample'] = BED_dataset['INFO'][:, 0].shape
        info['INFO']['exampleOfOneSample'] = BED_dataset['INFO'][:, 0]
        name_partsOfSpecDataset = ['Y', 'feat']
        for ind, name in enumerate(name_partsOfSpecDataset):
            info[name] = dict()
            info[name]['shapeOfAllSamples'] = dict()
            info[name]['shapeOfOneSample'] = dict()
            info[name]['exampleOfOneSample'] = dict()
        for indx, name in enumerate(name_partsOfSpecDataset):
            info[name]['shapeOfAllSamples'] = BED_dataset[name].shape
            info[name]['shapeOfOneSample'] = BED_dataset[name][0].shape
            info[name]['exampleOfOneSample'] = BED_dataset[name][0]

        # getting the number of samples for each subject
        labels = BED_dataset['Y']
        valueChanges = np.concatenate(
            (np.array([0]), np.unique(np.where(labels[:-1, ] != labels[1:, ])[0]), np.array([labels.shape[0] - 1])),
            axis=0)
        subjectSamples = []
        for i in range(0, len(valueChanges) + 1, 3):
            if i == 0:
                continue
            subjectSamples.append(valueChanges[i] - valueChanges[i - 3])
        numofSamplesPerSubject = dict()
        for m in range(len(subjectSamples)):
            numofSamplesPerSubject[str(m + 1)] = subjectSamples[m]
        info['numofSamplesPerSubject'] = numofSamplesPerSubject

        subjectSessionSamples = []
        for i in range(0, len(valueChanges)):
            if i == 0:
                continue
            subjectSessionSamples.append(valueChanges[i] - valueChanges[i - 1])
        numofSamplesPerSubjectSession = dict()
        for m in range(len(subjectSessionSamples)):
            numofSamplesPerSubjectSession[str(m + 1)] = subjectSessionSamples[m]
        info['numofSamplesPerSubjectSession'] = numofSamplesPerSubjectSession

        print("All elements within the SPEC dict: ", all_SPECdataset_names)
        print("Parts of the SPEC dataset        : ", info["all_parts_of_SPEC_dataset"])
        print("=================================")
        print("feat")
        print("    Shape of the features for all samples", info["feat"]['shapeOfAllSamples'])
        print("    Shape of the features for one sample ", info["feat"]['shapeOfOneSample'])
        print('Y')
        print("    Shape of the labels     ", info['Y']['shapeOfAllSamples'])
        print("    Example for one sample: ", info['Y']['exampleOfOneSample'])
        print("    Amount of samples for each subject     ")
        print("    ===========================================")
        for i in range(1, len(subjectSamples) + 1):
            print("         Subject#" + str(i), info['numofSamplesPerSubject'][str(i)])
        print('INFO')
        print("    Shape of the Info       ", info["INFO"]['shapeOfAllSamples'])
        print("    Shape of one sample:    ", info["INFO"]['shapeOfOneSample'])
        print("    Example for one sample: ", info["INFO"]['exampleOfOneSample'])

        return info, subjectSessionSamples
    def func_getOnlyGoodSubjectsData(self, X_train, X_test, Y_train, Y_test):
        indexesTrain = []
        indexesTest  = []
        goodSubjects = [10, 12, 13, 14, 4, 5, 7, 8, 9]

        for subject in goodSubjects:
            ind_train = list(Y_train[Y_train[0] == subject].index)
            ind_test  = list(Y_test[Y_test[0]   == subject].index)
            indexesTrain.extend(ind_train)
            indexesTest.extend(ind_test)


        Y_train = Y_train.loc[indexesTrain]
        Y_test  = Y_test.loc[indexesTest]
        X_train = X_train.loc[indexesTrain]
        X_test  = X_test.loc[indexesTest]

        return X_train, X_test, Y_train, Y_test

    def func_dataPreProcessing(self, BED_dataset, toCategorical, subjectRemoval):

        #GET THE FEATURES AND LABELS
        X_frame    = pd.DataFrame(BED_dataset['feat'])
        Y_frame    = pd.DataFrame(BED_dataset['Y'])

        #REMOVE ANY NANS
        ind        = pd.isnull(X_frame)
        no_nan     = X_frame.dropna().index.values
        X_frame    = X_frame.iloc[no_nan]
        Y_frame    = Y_frame.iloc[no_nan]

        #REPLACING THE LABELS TO MAKE THE FIRST TWO SESSIONS FOR TRAINING AND THE THIRD SESSION FOR TESTING
        Y_frame[1] = Y_frame[1].replace(2, 1)
        Y_frame[1] = Y_frame[1].replace(3, 0)
        ind_train  = list(Y_frame[Y_frame[1] == 1].index)
        ind_test   = list(Y_frame[Y_frame[1] == 0].index)

        #CREATING TRAIN/TEST DATASETS
        X_train    = X_frame.loc[ind_train]
        Y_train    = Y_frame[Y_frame[1] == 1]
        Y_train    = Y_train.drop(1, axis=1)

        X_test     = X_frame.loc[ind_test]
        Y_test     = Y_frame[Y_frame[1] == 0]
        Y_test     = Y_test.drop(1, axis=1)

        #IF CHOSEN OPTION "AFTER SUBJECT REMOVAL", THEN ONLY GET THE GOOD SUBJECTS
        if subjectRemoval == 'after subject removal':
            X_train, X_test, Y_train, Y_test = self.func_getOnlyGoodSubjectsData(X_train, X_test, Y_train, Y_test)

        Y_train = np.array(Y_train)
        Y_test  = np.array(Y_test)
        #TRANSFROM THE LABELS FROM INTEGERS TO ONE HOT VECTORS
        if toCategorical == 'true':
            Y_train = self.func_oneHotEncoder(Y_train, categorical='true')
            Y_test  = self.func_oneHotEncoder(Y_test, categorical='true')

        #SHUFFLING THE TRAINING DATASET
        X_train, Y_train = shuffle(X_train, Y_train, random_state=3)

        #STANDARDIZING THE DATASETS
        X_train, X_test = self.func_createStandardizedDataset(X_train, X_test, scaler_type="StandardScaler")

        return X_train, X_test, Y_train, Y_test

    def func_oneHotEncoder(self, dataset, categorical):
        if categorical == "true":
            dataset = to_categorical(dataset)
        elif categorical == "false":
            dataset = dataset
        return dataset

    def func_createStandardizedDataset(self, X_train, X_test, scaler_type):

        X_train = np.array(X_train)
        X_test  = np.array(X_test)

        if scaler_type == "StandardScaler":  # mean removal and variance scaling
            sc = StandardScaler()
            X_train, X_test = self.func_fit_transformScalertype(sc, X_train, X_test)

        elif scaler_type == "MinMaxScaler":  # Scaling features to a range
            mn = MinMaxScaler()
            X_train, X_test = self.func_fit_transformScalertype(mn, X_train, X_test)

        elif scaler_type == "MaxAbsScaler":  # scaling sparse data
            ma = MaxAbsScaler()
            X_train, X_test = self.func_fit_transformScalertype(ma, X_train, X_test)

        elif scaler_type == "Normalizer":  # normalizing the dataset
            nl = Normalizer()
            X_train, X_test = self.func_fit_transformScalertype(nl, X_train, X_test)

        elif scaler_type == "RobustScaler":  # scaling data with outliers
            rb = RobustScaler()
            X_train, X_test = self.func_fit_transformScalertype(rb, X_train, X_test)

        elif scaler_type == "QuantileTransformer":  # Mapping to a Uniform distribution
            qt = QuantileTransformer()
            X_train, X_test = self.func_fit_transformScalertype(qt, X_train, X_test)

        elif scaler_type == "PowerTransformer":  # Mapping to a Gaussian distribution
            pt = PowerTransformer()
            X_train, X_test = self.func_fit_transformScalertype(pt, X_train, X_test)

        return X_train, X_test

    def func_fit_transformScalertype(self, scalerType, X_train, X_test):
        scalerType.fit(X_train)
        X_train = scalerType.transform(X_train)
        X_test  = scalerType.transform(X_test)
        return X_train, X_test

    def func_createTensorDataset(self, X_train, X_test, Y_train, Y_test):

        #CREATES A TENSORDATASET FOR
        tensorXtr = torch.Tensor(X_train)  # transform to torch tensor
        tensorYtr = torch.Tensor(Y_train)
        my_dataset   = TensorDataset(tensorXtr, tensorYtr.long())
        train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=64, shuffle=True)

        tensorXte = torch.Tensor(X_test)  # transform to torch tensor
        tensorYte = torch.Tensor(Y_test)
        my_dataset2  = TensorDataset(tensorXte, tensorYte.long())
        test_loader  = torch.utils.data.DataLoader(my_dataset2, batch_size=64, shuffle=True)

        return train_loader, test_loader