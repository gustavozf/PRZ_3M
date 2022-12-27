import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from prz.dataset.egcz import EgczDataset
from prz.classification.cnn_fine_tuner import CnnFineTuner
from prz.utils.plot import plot_history

def update_out_dict(
        output_data: dict,
        group: int=0,
        file_paths: np.array=None,
        y_true: np.array=None,
        probas: np.array=None,
    ):
    output_data['file_path'].extend(file_paths)
    output_data['group_id'].extend([group for _ in range(len(y_true))])
    output_data['y_true'].extend(np.argmax(y_true, axis=1))
    output_data['y_pred'].extend(np.argmax(probas, axis=1))

    for i in range(probas.shape[1]):
        output_data[f'proba_{i}'].extend(probas[:, i])

def create_out_dict(n_classes: int):
    return {
        'file_path': [],
        'group_id': [],
        'y_true': [],
        'y_pred': [],
        **{
            f'proba_{i}': [] for i in range(n_classes)
        }
    }

def dump_json(data, out_path: str):
    with open(out_path, 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

def get_args():
    parser = argparse.ArgumentParser(description='Train with EGC-Z datasets.')

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=True,
        help='Input CSV file.'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='output/',
        help='Output path.'
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='ResNet50V2',
        choices=list(CnnFineTuner.MODELS.keys()),
        help='Base model for training.'
    )
    parser.add_argument(
        '-c',
        '--clf_layers',
        type=str,
        default='roeckernet',
        choices=list(CnnFineTuner.CLF_LAYERS.keys()),
        help='Classification layers configuration.'
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=64,
        help='Total number of epochs for training.'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=32,
        help='Batch size.'
    )

    return parser.parse_args()

def main():
    args = get_args()

    dataset = EgczDataset.from_csv(args.input)
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    gen_output = os.path.join(args.output, time_stamp)
    out_data = create_out_dict(dataset.n_classes)
    out_pred_path = os.path.join(gen_output, 'preds.csv')

    if not os.path.exists(gen_output):
        os.makedirs(gen_output)

    dump_json(vars(args), os.path.join(gen_output, 'args.json'))

    count = 0
    for train_index, test_index in dataset.leave_one_out_kfold_cv():
        X_train, X_test = dataset.data[train_index], dataset.data[test_index]
        y_train, y_test = dataset.label[train_index], dataset.label[test_index]
        group_id = dataset.groups[test_index][0]

        # Process the input data
        X_train = CnnFineTuner.MODELS[args.model].preprocess_data_array(
            X_train, resize=True
        )
        X_test = CnnFineTuner.MODELS[args.model].preprocess_data_array(
            X_test, resize=True 
        )

        print(f'Training fold #{count} / Testing group: {group_id}')
        group_output = os.path.join(gen_output, f'group_{group_id}')
        model_output = os.path.join(group_output, f'model.tf')

        if not os.path.exists(group_output):
            os.makedirs(group_output)

        hist, _ = CnnFineTuner.fine_tuning(
            X_train, y_train,
            X_test, y_test,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            clf_layers=args.clf_layers,
            n_classes=dataset.n_classes,
            loss='binary_crossentropy',
            callbacks=[
                ModelCheckpoint(
                    filepath=model_output,
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'
                )
            ],
        )

        # Get predictions
        model = load_model(model_output)
        probas = model.predict(X_test)
        
        # Update the output dict
        update_out_dict(
            out_data,
            group=group_id,
            y_true=y_test,
            probas=probas,
            file_paths=dataset.file_path[test_index]
        )

        # Save outputs
        pd.DataFrame(out_data).to_csv(out_pred_path)
        plot_history(hist.history, os.path.join(group_output, 'hist'))

        count += 1

        del X_train, X_test, y_train, y_test, hist
        K.clear_session()

    dump_json(
        classification_report(
            out_data['y_true'], out_data['y_pred'], digits=4, output_dict=True,
        ),
        os.path.join(gen_output, 'classification_report.json'),
    )

if __name__ == '__main__':
    main()