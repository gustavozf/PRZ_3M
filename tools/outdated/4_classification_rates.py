import sys
import os

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# The following line is needed if the lib is being accessed localy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PRZ_3M/')))

from prz.analysis.classification import accuracy_from_labels
from prz.resources.data.io import DataIO

# Consts
INPUT_PATH = 'D:\\Documents\\Mestrado\\Projeto\\Codes\\outputs\\'
EXT = '.csv'


OUTPUT_PATH = f'{INPUT_PATH}analysis\\'
OUTPUT_NAME = 'clf_rates'

FILE_NAMES = [
    'cqp_2_18',
    'eqp_2_18',
    'hqp_2_18',
    'inceptionv3',
    'lbp_2_8',
    'lpq_13',
    'pqp_2_18',
    'rlbp_2_8',
    'sqp_2_18',
    'vgg16',
]

img_types = [
    'EGC', 
    'ENC', 
    'MRG',
]

anim_tags = {
 '1C',
 '2C',
 '3C',
 '4C',
 '5C',
 '6C',
 '7C',
 '22TW',
 '23TW',
 '24TW',
 '26TW',
 '28TW',
 '30TW',
}

accuracies = {
    j: [] for j in list(anim_tags) + [
        'img_type', 
        'extractor', 
        'mean', 
        'median', 
        'std_dev', 
        'variance',
        'f1_score',
        'precision',
        'recall',
        'accuracy',
    ]
}

DataIO.createDirIfNotExists(OUTPUT_PATH)

for count, file_name in zip(range(len(FILE_NAMES)), FILE_NAMES):
    print(count, file_name)
    curr_input_path = f'{INPUT_PATH}predictions\\{file_name}\\'

    for img_type in img_types:
        accuracies['img_type'].append(img_type)
        accuracies['extractor'].append(file_name)

        aux_acc = []
        aux_y_label = []
        aux_y_pred = []

        for anim_tag in anim_tags:
            label = 0 if anim_tag.endswith('C') else 1

            curr_file = (
                f'{curr_input_path}{anim_tag}\\{img_type}\\' +
                f'{anim_tag}_{img_type}_test.csv.predict'
            )
            
            df = pd.read_csv(curr_file, sep=' ', header=0)
            
            assert len(df['labels']) == 32

            acc = accuracy_from_labels(
                    y_pred=df['labels'],
                    y_label=[label for i in range(32)]
                )

            aux_y_label.extend([label for i in range(32)])
            aux_y_pred.extend(df['labels'])

            accuracies[anim_tag].append(acc)
            aux_acc.append(acc)

            del df, curr_file, label, acc
        
        accuracies['mean'].append(np.mean(aux_acc))
        accuracies['median'].append(np.median(aux_acc))
        accuracies['std_dev'].append(np.std(aux_acc))
        accuracies['variance'].append(np.var(aux_acc))

        report = classification_report(aux_y_label, aux_y_pred, digits=4, output_dict=True)

        accuracies['f1_score'].append(report['weighted avg']['f1-score'])
        accuracies['precision'].append(report['weighted avg']['precision'])
        accuracies['recall'].append(report['weighted avg']['recall'])
        accuracies['accuracy'].append(report['accuracy'])

        del aux_acc, aux_y_label, aux_y_pred, report

print('Printing to file...')
df = pd.DataFrame(data=accuracies)

df.to_csv(
    f'{OUTPUT_PATH}{OUTPUT_NAME}.csv', 
    sep=',',
    header=True,
    index=False,
)

df.applymap(lambda x : str(x).replace('.', ',')).to_csv(
    f'{OUTPUT_PATH}{OUTPUT_NAME}_br.csv', 
    sep=';',
    header=True,
    index=False,
)