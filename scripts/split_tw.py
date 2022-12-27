import os
import shutil

import pandas as pd

from utils import walk, split_path

BASE_PATH = os.path.abspath(os.path.dirname(__file__))
INPUT_PATH = os.path.join(BASE_PATH, '..', 'data', 'raw', 'TW')
OUTPUT_PATH = os.path.join(BASE_PATH, '..', 'data', 'interim')
EXPECTED_NUM_SAMPLES = 1248

mrg_data = []
enc_data = []
egc_data = []

# get the animal tags and enumerate them
animal_tags = {
    tag : _id
    for _id, tag in enumerate(os.listdir(INPUT_PATH))
}
# list all of the JPG files
file_paths, count = walk(INPUT_PATH, target_file_ext='JPG')

assert count == EXPECTED_NUM_SAMPLES

# split the data accordingly to the visualization type
print('Splitting data...')
for _file in file_paths:
    if 'c1+c2' in _file:
        mrg_data.append(_file)
    elif 'c1' in _file:
        enc_data.append(_file)
    else:
        egc_data.append(_file)

assert len(mrg_data) + len(enc_data) + len(egc_data) == EXPECTED_NUM_SAMPLES

        
if not os.path.exists(OUTPUT_PATH):
    print('Creating path: ', OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

print('Generating outputs...')
iter_data =  [('mrg', mrg_data), ('enc', enc_data), ('egc', egc_data)]
for _name, data_list in iter_data:
    out_data = {
        'file_path': [],
        'anim_tag': [],
        'anim_tag_id': [],
        'sample_id': [],
        'label': [],
    }

    for data_path in data_list:
        animal_tag, sample_id = split_path(data_path)[-3:-1]

        out_data['file_path'].append(data_path)
        out_data['anim_tag'].append(animal_tag)
        out_data['anim_tag_id'].append(animal_tags[animal_tag])
        out_data['sample_id'].append(int(sample_id))
        out_data['label'].append(0 if 'C' in animal_tag else 1)

    out_path = os.path.join(OUTPUT_PATH, f'tw_{_name}.csv')
    print('Saving file: ', out_path)
    pd.DataFrame(out_data).to_csv(out_path)