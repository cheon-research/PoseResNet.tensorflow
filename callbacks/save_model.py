import numpy as np
import keras.api._v2.keras as keras

import os


class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, path, now_str, monitor='val_loss'):
        self.path = path
        self.now_str = now_str
        self.monitor = monitor
        self.best_loss = 99
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs[self.monitor] < self.best_loss:
            self.best_loss = logs['val_loss']
            self.best_epoch = epoch
            model_fname = f'{self.path}model_{self.now_str}'
            self.model.save(model_fname)
            print('    [SaveModelCallback] best model saved.')
