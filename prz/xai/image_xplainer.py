import os

import shap
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from skimage.segmentation import mark_boundaries

SHAP_XPLAINERS = {
    'DeepExplainer': shap.DeepExplainer,
    'GradientExplainer': shap.GradientExplainer,
}

def preprocess_lime(x):
    return x / 2 + 0.5

def fixed_input_model(model, input_shape: tuple):
    new_input = Input(batch_shape=input_shape)
    new_outputs = model(new_input)
    new_model = Model(new_input,new_outputs)

    new_model.set_weights(model.get_weights())

    return new_model

def get_shap_xplainer(name: str):
    assert name in set(SHAP_XPLAINERS.keys()), 'Invalid SHAP explainer!'

    return SHAP_XPLAINERS[name]

class ImageXplainer:
    @staticmethod
    def shap(
            pred_fn,
            data: np.array,
            shape: np.array,
            class_names: list = ['C', 'TW'],
            n_evals: int = 50000,
            masker_mode: str = 'inpaint_telea'):
        explainer = shap.Explainer(
            pred_fn,
            masker=shap.maskers.Image(masker_mode, shape),
            output_names=class_names)

        return explainer(
            data,
            max_evals=n_evals,
            batch_size=50,
            outputs=shap.Explanation.argsort.flip[:2]
        )
    
    @staticmethod
    def shap_deep(
            model,
            data: np.array,
            background: np.array,
            batch_size: int = 50,
            xplainer: str = 'DeepExplainer'):
        model = fixed_input_model(model, (batch_size,) + background.shape[1:])
        e = get_shap_xplainer(xplainer)(
            model, background, batch_size=batch_size
        )
        return e.shap_values(data)
        # return e(
        #     data[:4],
        #     max_evals=50000,
        #     batch_size=50,
        #     outputs=shap.Explanation.argsort.flip[:2]
        # )
        
    @staticmethod
    def lime(
            pred_fn,
            data: np.array,
            labels: np.array,
            num_samples: int = 4048,
            process_img_fn = preprocess_lime):
        """
            Source from: https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html
        """
        output_img_list = []
        explainer = lime_image.LimeImageExplainer()

        for i in range(len(data)):
            plt.clf()
            image = data[i]
            label = labels[i]
            xplnation  = explainer.explain_instance(
                image,
                pred_fn,
                batch_size=50,
                num_samples=num_samples,
                random_seed=101)
            
            temp, mask = xplnation.get_image_and_mask(
                label,
                positive_only=False,
                num_features=5,
                hide_rest=False,
            )

            output_img_list.append(
                mark_boundaries(process_img_fn(temp), mask)
            )
        
        return output_img_list

    @staticmethod
    def plot_lime(
            imgs: list,
            out_path: str,
            img_names: list):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for i in range(len(imgs)):
            plt.imshow(imgs[i])
            plt.savefig(os.path.join(out_path, img_names[i]))
            plt.clf()

    @staticmethod
    def plot_shap(
            imgs: np.array,
            shap_values: np.array,
            out_path: str,
            img_names: list):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        print(shap_values.shape)

        for i in range(len(shap_values)):
            # shap_numpy = [
            #     np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values
            # ]
            # test_numpy = np.swapaxes(np.swapaxes(imgs, 1, -1), 1, 2)

            # shap.image_plot(shap_numpy, -test_numpy)

            print(shap_values[i].shape)
            print(imgs[i].shape)

            shap.image_plot(shap_values[i], show=False)
            plt.savefig(os.path.join(out_path, img_names[i]))
            plt.clf()

            return
