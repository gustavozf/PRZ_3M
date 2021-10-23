import os
import json
import argparse
from datetime import datetime

from numpy import argmax
from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model 

from prz.dataset.egcz import EgczDataset
from prz.classification.cnn_fine_tuner import CnnFineTuner
from prz.utils.plot import plot_history

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
        choices={
            'VGG16',
            'VGG19',
            'ResNet50V2',
            'ResNet101V2',
            'ResNet152V2',
            'InceptionV3',
            'InceptionResNetV2',
            'MobileNetV2'
        },
        help='Base model for training.'
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

    if not os.path.exists(gen_output):
        os.makedirs(gen_output)

    dump_json(vars(args), os.path.join(gen_output, 'args.json'))

    out_data = {
        'y_true': [],
        'y_probas': [],
        'y_pred': [],
    }
    count = 0
    for train_index, test_index in dataset.leave_one_out_kfold_cv():
        X_train, X_test = dataset.data[train_index], dataset.data[test_index]
        y_train, y_test = dataset.label[train_index], dataset.label[test_index]
        group_id = dataset.groups[test_index][0]

        # Process the input data
        X_train = CnnFineTuner.preprocess_data_array(
            X_train, args.model, model.input_shape 
        )
        X_valid = CnnFineTuner.preprocess_data_array(
            X_valid, args.model, model.input_shape 
        )

        print(f'Training fold #{count} / Testing group: {group_id}')
        group_output = os.path.join(gen_output, f'group_{group_id}')
        # model_output = os.path.join(
        #     group_output,
        #     f'model_{group_id}_' + '{epoch:02d}-{val_loss:.2f}.hdf5'
        # )

        if not os.path.exists(group_output):
            os.makedirs(group_output)

        hist, model = CnnFineTuner.fine_tuning(
            X_train, y_train,
            X_test, y_test,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss='binary_crossentropy',
            callbacks=[
                # ModelCheckpoint(
                #     filepath=model_output,
                #     save_best_only=True,
                #     monitor='val_accuracy',
                # ),
                # TensorBoard(log_dir=os.path.join(curr_output, 'logs')),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=10,
                    min_lr=0.000001
                )
            ],
        )

        plot_history(hist.history, os.path.join(group_output, f'{count}_hist'))

        # Get predictions
        # model = load_model(model_output)
        probas = model.predict(X_test)

        out_data['y_true'].extend(y_test)
        out_data['y_probas'].extend(probas)
        out_data['y_pred'].extend(argmax(probas))
        dump_json(out_data, os.path.join(gen_output, 'preds.json'))

        count += 1

        del X_train, X_test, y_train, y_test, hist
        K.clear_session()

if __name__ == '__main__':
    main()