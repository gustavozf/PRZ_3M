# imports
import sys
import os
import pandas as pd

# CONSTS
INPUT_PATH = 'D:\\Documents\\Mestrado\\Projeto\\Codes\\outputs\\'
FILE_NAME = 'inceptionv3'
EXT = '.csv'

FEATURE_INPUT_PATH = f'{INPUT_PATH}features\\svm_format\\{FILE_NAME}\\'
PREDICTION_OUTPUT_PATH = f'{INPUT_PATH}predictions\\{FILE_NAME}\\'

LIBSVM_EASY = 'D:\\Documents\\Software\\libsvm-3.24\\tools\\easy.py'

# VARS
img_types = [
    'EGC', 
    'ENC', 
    'MRG',
]

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
if not os.path.exists(PREDICTION_OUTPUT_PATH):
    os.makedirs(PREDICTION_OUTPUT_PATH)

for anim_tag in anim_tags:
    for img_type in img_types:

        curr_tag = f'{anim_tag}_{img_type}'
        out_file = (PREDICTION_OUTPUT_PATH + 
            f'{curr_tag}_pred{EXT}')

        test_file = f'{FEATURE_INPUT_PATH}{curr_tag}_test{EXT}'
        train_file = f'{FEATURE_INPUT_PATH}{curr_tag}_train{EXT}'

        print('Classifying: ', anim_tag, img_type)
        os.system(f'python {LIBSVM_EASY} {train_file} {test_file}')

        print('Moving Files')
        print()

        generated_files = [
            f'{anim_tag}_{img_type}_test{EXT}.predict',
            f'{anim_tag}_{img_type}_test{EXT}.scale',
            f'{anim_tag}_{img_type}_train{EXT}.model',
            f'{anim_tag}_{img_type}_train{EXT}.range',
            f'{anim_tag}_{img_type}_train{EXT}.scale',
            f'{anim_tag}_{img_type}_train{EXT}.scale.out',
        ]

        curr_out_path = f'{PREDICTION_OUTPUT_PATH}{anim_tag}\\{img_type}\\'
        if not os.path.exists(curr_out_path):
            os.makedirs(curr_out_path)

        for generated_file in generated_files:
            os.rename(generated_file, curr_out_path + generated_file)

        del curr_tag, out_file
        del test_file, train_file
        del generated_files, curr_out_path

