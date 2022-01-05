import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from prz.dataset.egcz import EgczDataset
from prz.classification.cnn_fine_tuner import CnnFineTuner
from prz.xai.image_xplainer import ImageXplainer

def make_output_path(out_path: str, img_name: str):
    return os.path.join(
        out_path,
        os.path.basename(img_name).split('.')[1] + '.svg'
    )

def get_args():
    parser = argparse.ArgumentParser(
        description='Train XAI models to explain the models '
                    'developed with the EGC-Z dataset.'
    )

    parser.add_argument(
        '-i',
        '--input_path',
        type=str,
        required=True,
        help='Path to the test data in a CSV format.'
    )
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model.'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='output/',
        help='Output path.'
    )
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='ResNet50V2',
        choices=list(CnnFineTuner.MODELS.keys()),
        help='Base model for training.'
    )
    parser.add_argument(
        '-t',
        '--tag',
        type=str,
        default='TW',
        choices={'TW', 'D', 'AIA'},
        help='Disease tag.'
    )

    return parser.parse_args()

def main():
    args = get_args()

    print('Loading data')
    dataset = EgczDataset.from_csv(args.input_path)
    print('Preprocessing data')
    X_data = CnnFineTuner.MODELS[args.model_name].preprocess_data_array(
        dataset.data, resize=True
    )
    print('Loading model')
    model = load_model(args.model_path)

    print('Xplaining data')
    xplainer = ImageXplainer(
        X_data,
        np.argmax(dataset.label, axis=1),
        model.predict,
        class_names=['C', args.tag])

    shap_out_list = xplainer.shap()
    lime_out_list = xplainer.lime()
    
    print('Generating outputs')
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for i in range(len(dataset.data)):
        xplainer.plot(
            make_output_path(args.output, dataset.file_path[i]),
            dataset.label[i],
            X_data[i]/2 + 0.5,
            np.zeros(X_data[i].shape),
            lime_out_list[i]
        )

if __name__ == '__main__':
    main()