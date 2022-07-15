# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-07-23 10:56:37
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-10-03 13:51:28

import matplotlib; matplotlib.use('agg')
import keras
import numpy as np
import matplotlib.pyplot as plt
import os

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
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.num_epochs = np.arange(0, len(self.losses))
        self.make_plots()

    def make_plots(self):

        self.make_plot(
            self.losses,
            self.val_losses,
            "Loss",
            "plots/loss_plot.png"
        )

        self.make_plot(
            self.acc,
            self.val_acc,
            "Accuracy",
            "plots/accuracy_plot.png"
        )

    def make_plot(self, train_metric, val_metric, ylabel, save_path):

        plt.style.use("seaborn")
        plt.figure()

        plt.plot(self.num_epochs, train_metric, label="training")
        plt.plot(self.num_epochs, val_metric, label="validation")

        plt.title(f"Training and Validation {ylabel}")

        plt.xlabel("Epoch #")
        plt.ylabel(ylabel)
        
        plt.legend()
        plt.savefig(save_path)

        plt.close()
