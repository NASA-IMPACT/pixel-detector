# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-07-23 10:56:37
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-09-30 16:12:51

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import keras
import numpy as np


class TrainingPlot(keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        self.num_epochs = np.arange(0, len(self.losses))

    def loss_plot(self, epoch):
        plt.style.use("seaborn")
        plt.figure()
        plt.plot(self.num_epochs, self.losses, label="train_loss")
        plt.plot(self.num_epochs, self.val_losses, label="val_loss")
        plt.title("Training and Validation Loss".format(epoch))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('loss_plot_2.png'.format(epoch))
        plt.close()

        plt.figure()
        plt.plot(self.num_epochs, self.acc, label="train_acc")
        plt.plot(self.num_epochs, self.val_acc, label="val_acc")
        plt.title("Training and Validation Accuracy".format(epoch))
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('acc_plot_2.png'.format(epoch))
        plt.close()
