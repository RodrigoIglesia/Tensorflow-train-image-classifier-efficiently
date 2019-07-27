'''
Helper functions to show training state and results.
Author: Rodrigo de la Iglesia.
Version: 1.0.
27/07/2019
'''
import tensorflow as tf
import numpy as np
import psutil
import math
import sys
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def checkRAM():
    ram_used = psutil.virtual_memory()[2]
    if ram_used > 60.0:
        sys.exit("Exceeded RAM usage.")

def plotResults(n, v_train, v_val, loss=True, title=None):
    plt.plot(range(len(v_train)), v_train, 'b', label='Training')
    plt.plot(range(len(v_train)), v_val, 'r', label='Validation')

    if title is not None:
        plt.title(title)

    plt.xlabel('Epochs',fontsize=16)

    if loss == True:
        plt.ylabel('Loss',fontsize=16)
    else:
        plt.ylabel('Accuracy',fontsize=16)

    plt.legend()
    plt.figure(n)
    plt.show()
    
def plotActivations(n, act):
    filters = act.shape[3]
    plt.figure(n, figsize=(20, 10)).subplots_adjust(
        left=0.01, bottom=0.01, right=1, top=1, wspace=0.1, hspace=0.1)
    cols = 5
    rows = math.ceil(filters / cols) + 1
    
    for i in range(filters):
        plt.subplot(rows, cols, i+1)
        plt.imshow(act[0,:,:,i], interpolation="nearest", aspect='equal', cmap='gray')
        plt.colorbar()

def showProgress(epoch, iteration, n_steps, loss, acc):
    sys.stdout.write('\r'+'Epoch: {0}...'.format(epoch)+str(iteration)+'/'+str(n_steps)+
                    '\tLoss: {:.4g}'.format(loss)+
                    '\tAccuracy: {:.2f}%'.format(acc*100))
    sys.stdout.flush() 

def showEpochResults(mean_loss, mean_acc):
    print("\nMean epoch loss: {:.4g}".format(mean_loss))
    print("Mean epoch accuracy: {:.2f}%".format(mean_acc*100))

    
