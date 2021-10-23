import json

import matplotlib.pyplot as plt

def plot_history(history: dict, out_path: str, save_json: bool=False):
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


    if save_json:
        with open(out_path+'_hist.json', 'w') as json_file:
            json.dump(history, json_file, indent=2)