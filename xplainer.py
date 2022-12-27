import os
import argparse

import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model

from prz.dataset.egcz import EgczDataset
from prz.classification.cnn_fine_tuner import CnnFineTuner
from prz.xai.image_xplainer import ImageXplainer

def xplain_model(
        model,
        model_name: str,
        X_data: np.array,
        labels: np.array,
        class_names: list = ['C', 'TW'],
        background: np.array = np.array([])):
    # shap_values = ImageXplainer.shap(
    #     model.predict,
    #     X_data,
    #     X_data[0].shape,
    #     class_names=class_names)

    shap_values, _ = ImageXplainer.shap_deep(
        model,
        X_data,
        background,
        batch_size=50,
        xplainer='GradientExplainer'
    )

    # lime_output = ImageXplainer.lime(
    #     model.predict,
    #     X_data,
    #     labels)

    lime_output = []
    return shap_values, lime_output

def get_random_sample(dataset: EgczDataset, group_id: int, model_name: str):
    background = dataset.data[get_indexes(dataset.groups != group_id)]
    rand_background = (background[
        np.random.choice(background.shape[0], 100, replace=False)
    ])

    return CnnFineTuner.MODELS[model_name].preprocess_data_array(
        background, resize=True
    )

def get_model(model_path: str):
    model_dir = list(
        filter(lambda x : x.endswith('.tf'), os.listdir(model_path))
    )[0]
    return load_model(os.path.join(model_path, model_dir))

def get_indexes(filter):
    idxs = np.argwhere(filter)
    return idxs.reshape(idxs.shape[0])

def get_args():
    parser = argparse.ArgumentParser(
        description='Train XAI models to explain the models '
                    'developed with the EGC-Z dataset.'
    )

    parser.add_argument(
        '-i',
        '--input_data',
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
    dataset = EgczDataset.from_csv(args.input_data)

    for group_id in tqdm(np.unique(dataset.groups)):
        full_model_path = os.path.join(args.model_path, f'group_{group_id}')
        idxs = get_indexes(dataset.groups == group_id)
        model = get_model(full_model_path)

        X_data = CnnFineTuner.MODELS[args.model_name].preprocess_data_array(
            dataset.data[idxs], resize=True
        )
        X_data = X_data[:5]
        
        shap_out_list, lime_out_list = xplain_model(
            model,
            args.model_name,
            X_data,
            np.argmax(dataset.label[idxs], axis=1),
            class_names=['C', args.tag],
            background=get_random_sample(dataset, group_id, args.model_name)
        )

        img_names = list(
            map(lambda x : os.path.basename(x), dataset.file_path[idxs])
        )

        output_path = os.path.join(full_model_path, 'xai')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # ImageXplainer.plot_lime(
        #     lime_out_list,
        #     os.path.join(output_path, 'lime'),
        #     img_names)

        ImageXplainer.plot_shap(
            X_data,
            shap_out_list,
            os.path.join(output_path, 'shap'),
            img_names)
        quit()

if __name__ == '__main__':
    main()