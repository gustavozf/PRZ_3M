# imports
import sys
import os
import pandas as pd

# CONSTS
INPUT_PATH = 'D:\\Documents\\Mestrado\\Projeto\\Codes\\outputs\\features\\'
FILE_NAME = 'inceptionv3'
EXT = '.csv'
OUTPUT_PATH = f'{INPUT_PATH}svm_format\\{FILE_NAME}\\'

# support variables
img_types = ['EGC', 'ENC', 'MRG']
anim_tags = {
 '1C',
 '22TW',
 '23TW',
 '24TW',
 '26TW',
 '28TW',
 '2C',
 '30TW',
 '3C',
 '4C',
 '5C',
 '6C',
 '7C'
}

# Create path if it doesn't exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# import
df = pd.read_csv(INPUT_PATH+FILE_NAME+EXT, sep=',', header=None)
# get last column containing the labels
keys = df.iloc[:,-1:].to_numpy()
# get the features
values = df.iloc[:,:-1].to_numpy()

del df

test_files = { 
    i : { 
        j : open(f'{OUTPUT_PATH}{i}_{j}_test{EXT}', 'w') for j in img_types 
    } 
    for i in anim_tags 
}

train_files = { 
    i: { 
        j : open(f'{OUTPUT_PATH}{i}_{j}_train{EXT}', 'w') for j in img_types 
    } 
    for i in anim_tags 
}

# read the information
for file_label, features in zip(keys, values):
    file_name = file_label[0].split('.png')[0]
    anim_tag, _, img_type = file_name.strip().split('_')
    label = 0 if anim_tag.endswith('C') else 1

    print(anim_tag, img_type)

    # Connvert the data to the LibSVM format
    count = 1
    svm_feat = f'{label} '
    for value in features:
        svm_feat += f'{count}:{value} '
        count += 1
    
    svm_feat = svm_feat[:-1] + '\n'

    # Write the data
    test_files[anim_tag][img_type].write(svm_feat)
    
    for tag in anim_tags - {anim_tag}:
        train_files[tag][img_type].write(svm_feat)

    # clear the vars
    del file_name, anim_tag, img_type
    del label, count, svm_feat

# 
for i in anim_tags:
    for j in img_types:
        test_files[i][j].close()
        train_files[i][j].close()