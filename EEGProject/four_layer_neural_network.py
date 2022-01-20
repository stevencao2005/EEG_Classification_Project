
"""
@author: Steven Cao"""

#IMPORT ALL NEEDED MODULES

#Standard library imports
import numpy as np
import os
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt

#Third party imports
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim



class Net(nn.Module):
    def __init__(self, epochs=100, in_features=224):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(in_features, 124)
        self.linear2 = nn.Linear(124, 64)
        self.linear3 = nn.Linear(64, 22)

        self.epochs  = epochs
        self.cost    = []

    def forward_pass(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def train(self, train_loader, optimizer, criterion):
        start_time = time.time()
        for iteration in range(self.epochs):
            one_loss    = []
            for batch_idx, (x, y) in enumerate(train_loader):
                ylabel = np.argmax(y, axis=1)

                #FORWARD PROPAGATION
                output = self.forward_pass(x)

                #BACKWARD PROPAGATION
                loss = criterion(output, ylabel)
                loss.backward()
                one_loss.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()

            all_loss = sum(one_loss)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Loss: {2}'.format(iteration + 1, time.time() - start_time, all_loss))
            self.cost.append(all_loss)
        pass

    def predict(self, test_loader):
        y_pred   = []
        y_true   = []
        for x, y in test_loader:

            output  = self.forward_pass(x).detach().numpy()

            y_pred.extend(np.argmax(output, axis=1))
            y_true.extend(np.argmax(y.numpy(), axis=1))

        return y_pred, y_true

    def compute_performance_metrics(self, test_loader):

        #PREDICTING THE TEST SAMPLES
        start_time     = time.time()
        y_pred, y_true = self.predict(test_loader)
        print("time took to go through the test dataset", time.time()-start_time)

        #EVALUTING THE MODEL
        confusionMatrix      = confusion_matrix(y_true, y_pred)

        metrics              = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                           columns=['accuracy', 'precision', 'recall', 'f1 score'])

        labels               = list(np.unique(y_true))
        metrics['accuracy']  = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, labels=labels, average='macro')
        metrics['recall']    = recall_score(y_true, y_pred, labels=labels, average='macro')
        metrics['f1 score']  = f1_score(y_true, y_pred, labels=labels, average='macro')

        return y_pred, y_true, metrics, confusionMatrix


    def plot_loss(self):
        plt.figure()
        plt.plot(self.cost)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.title("Cost curve for training")
        plt.show()