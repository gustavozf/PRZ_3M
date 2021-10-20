import os
import json
import argparse
from datetime import datetime

from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)

from prz.classification.pre_trained_cnn import PreTrainedCnnModels
from prz.dataset.egcz import EgczDataset

def dump_args(args, out_path: str):
    out_file = os.path.join(out_path, 'args.json')
    with open(out_file, 'w', encoding='utf8') as json_file:
        json.dump(vars(args), json_file, ensure_ascii=False, indent=2)

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
            'VGG16', 'VGG19', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
            'InceptionV3', 'InceptionResNetV2', 'MobileNetV2'
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
    curr_output = os.path.join(args.output, time_stamp)

    if not os.path.exists(curr_output):
        os.makedirs(curr_output)

    dump_args(args, curr_output)

    for train_index, test_index in dataset.leave_one_out_kfold_cv():
        X_train, X_test = dataset.data[train_index], dataset.data[test_index]
        y_train, y_test = dataset.label[train_index], dataset.label[test_index]

        PreTrainedCnnModels.fine_tuning(
            X_train, y_train,
            X_test, y_test,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss='binary_crossentropy',
            last_layer='sigmoid',
            callbacks=[
                ModelCheckpoint(
                    filepath=os.path.join(curr_output, 'model.h5'),
                    save_best_only=True,
                ),
                TensorBoard(log_dir=os.path.join(curr_output, 'logs')),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=20,
                    min_lr=0.000001
                )
            ],
        )


if __name__ == '__main__':
    main()