import numpy as np
import keras.api._v2.keras as keras
from matplotlib import pyplot as plt


def plot_model(model, file_name):
    keras.utils.plot_model(
        model.build_graph(),
        to_file=file_name,
        show_shapes=True,
        show_layer_names=True,
        show_dtype=True,
        rankdir='BT',
        expand_nested=False)


def plot_loss(history, start_epoch=1, now_str=''):
    fig, loss_ax = plt.subplots()
    #acc_ax = loss_ax.twinx()

    loss_ax.set_title(f'start from epoch {start_epoch}')

    loss_ax.plot(history.history['loss'][start_epoch:], 'b', label = 'train loss')
    loss_ax.plot(history.history['val_loss'][start_epoch:], 'r', label = 'val loss')

    #acc_ax.plot(hist.history['accuracy'], 'b', label = 'train accuracy')
    #acc_ax.plot(hist.history['val_accuracy'], 'g', label = 'val accuracy')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    #acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc = 'upper left')
    #acc_ax.legend(loc = 'lower left')

    print(f'history_{now_str}.png')
    plt.savefig(f'../results/history_{now_str}.png')
    

def lr_scheduler(epoch, lr):
    if epoch == 0:
        return lr
    elif epoch % 30 == 0:
        print(f'lr decay. from {lr} to {lr *0.5}')
        return lr * 0.5
    else:
        return lr
