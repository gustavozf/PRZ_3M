import collections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_history(history: dict, out_path: str, save_df: bool=True):
    #taken from: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(out_path + '_acc.png')
    plt.clf()

    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(out_path + '_loss.png')
    plt.clf()

    if save_df:
        pd.DataFrame(history).to_csv(out_path+'_hist.csv')

def plot_acc_comparison(
        data: dict,
        out_path: str,
        labels: list,
        width: float = 0.88,
        data_keys: list = ['CGE', 'CNE', 'MRG'],
        colors: list = [
            '#FDD6D3', # (253, 214, 211)
            '#B3EAC3', # (179, 234, 195)
            '#D0E1FF', # (208, 225, 255)
        ],
        ylabel: str = 'Animal',
        xlabel: str = 'Acurácia'):
    data_keys_len = len(data_keys)
    assert data_keys_len > 0
    assert data_keys_len == len(colors)

    df = pd.DataFrame(data, index=labels)
    _, ax2 = plt.subplots()

    x_lim = (min(df.min()) - 0.01, max(df.max()) + 0.001)
    df.plot.barh(
        color={data_keys[i]:colors[i] for i in range(data_keys_len)},
        ax=ax2,
        xlim=x_lim,
        width=width, 
        figsize=(8,6))

    ax2.invert_yaxis()  

    for container in ax2.containers:
        ax2.bar_label(container, padding=-20, fmt='%.3f', fontsize=6 )
    
    ax2.grid(axis='x', linestyle='--')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)

    plt.legend(loc='upper left')
    plt.savefig(out_path + 'acc_comp.png')
    plt.clf()
