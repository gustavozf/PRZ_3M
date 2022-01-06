import shap
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

class ImageXplainer:

    def __init__(
            self,
            data: np.array,
            labels: np.array,
            pred_fn,
            class_names: list = ['C', 'TW']) -> None:
        self.data = data
        self.labels = labels
        self.shape = data[0].shape
        self.pred_fn = pred_fn
        self.class_names = class_names

    def shap(self, n_evals: int = 500):
        explainer = shap.Explainer(
            self.pred_fn,
            shap.maskers.Image("inpaint_telea", self.shape),
            output_names=self.class_names)
        shap_values = explainer(
            self.data,
            max_evals=n_evals,
            outputs=shap.Explanation.argsort.flip[:2]
        )
        print(shap_values.shape)
        print(shap_values[0].shape)
        print(self.labels.shape)
        print(np.tile(np.array(shap_values.output_names), 2))

        for i in range(shap_values.shape[0]):
            shap.image_plot(shap_values[i])
            input('Inserir qlqr coisa')

    def lime(
            self,
            num_samples: int = 2024,
            process_img_fn = lambda x : x / 2 + 0.5):
        """
            Source from: https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html
        """
        output_img_list = []
        explainer = lime_image.LimeImageExplainer()

        for i in range(len(self.data)):
            plt.clf()
            image = self.data[i]
            label = self.labels[i]
            xplnation  = explainer.explain_instance(
                image,
                self.pred_fn,
                batch_size=50,
                num_samples=num_samples)
            
            temp, mask = xplnation.get_image_and_mask(
                label,
                positive_only=False,
                num_features=5,
                hide_rest=False
            )

            output_img_list.append(
                mark_boundaries(process_img_fn(temp), mask)
            )
        
        return output_img_list

    def plot(
            self,
            output_path: str,
            true_label: int,
            image: np.array,
            shap_output: np.array,
            lime_output: np.array,
            ):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('Image Explanation')
        ax1.imshow(image)
        ax2.imshow(shap_output)
        ax3.imshow(lime_output)

        ax1.set_title('Original')
        ax2.set_title('SHAP')
        ax3.set_title('Lime')

        plt.axis('off')
        plt.savefig(output_path, format="svg")
        plt.clf()
