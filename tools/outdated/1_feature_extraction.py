import sys
import os
import re
import platform
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image

# The following line is needed if the lib is being accessed localy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PRZ_3M/')))

from prz.feature_extraction.nhc.transfer_learning import CNNPredictor, PreTrainedCNNModels
from prz.resources.data.io import DataIO

def nhc_extraction(input_path='', out_name='out', predictor=None):
    count = 1

    if not predictor:
        return

    with open(out_name+'.csv', 'w') as out_file:
        for img_name in DataIO.listDir(input_path):
            print(count, img_name)
        
            img = image.load_img(
                input_path + img_name, 
                target_size=predictor.model.min_input_shape[:2],
            )
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = predictor.model.preprocess_input(img_data)

            X = predictor.extract_features(imgs=img_data).flatten()

            del img, img_data
            count += 1

            for feature in X:
                out_file.write(f'{feature}, ')

            out_file.write(f'{img_name}\n')

            
if __name__ == "__main__":
    INPUT_PATH = 'D:\\Documents\\Mestrado\\Projeto\\Codes\\outputs\\dataset_gs\\'

    vgg16 = CNNPredictor(model=PreTrainedCNNModels.VGG16(input_shape='min'))
    vgg16.summary()

    # inceptionV3 = CNNPredictor(model=PreTrainedCNNModels.InceptionV3(input_shape='min'))
    # inceptionV3.summary()

    nhc_extraction(
        input_path=INPUT_PATH, 
        out_name='../outputs/features/vgg16', 
        predictor=vgg16
    )
