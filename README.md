# PR-Z_3M

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository makes available the source code used throught the development of the work described in [Cancer Identification in Enteric Nervous System Preclinical Images Using Handcrafted and Automatic Learned Features](https://doi.org/10.1007/s11063-022-11114-y), published to Neural Processing Letters.

## Abstract
Chronic degenerative diseases affect Enteric Neuron Cells (ENC) and Enteric Glial Cells (EGC) in shape and quantity. Thus, searching for automatic methods to evaluate when these cells are affected is quite opportune. In addition, preclinical imaging analysis is outstanding because it is non-invasive and avoids exposing patients to the risk of death or permanent disability. We aim to identify a specific cancer experimental model (Walker-256 tumor) in the Enteric Nervous System (ENS) cells. The ENS image database used in our experimental evaluation comprises 1248 images taken from thirteen rats distributed in two classes: control/healthy or sick. The images were created with three distinct contrast settings targeting different ENS cells: ENC, EGC, or both. We extracted handcrafted and non-handcrafted features to provide a comprehensive classification approach using SVM as the core classifier. We also applied Late Fusion techniques to evaluate the complementarity between feature sets obtained in different scenarios. In the best case, we achieved an F1-score of 0.9903 by combining classifiers built from different image types (ENC and EGC), using Local Phase Quantization (LPQ) features.

## Project Organization
The overall data organization is described as follows:

- `train.py`: main script for training the deep learning models
- `clf_rates.py`: script used for generating the classification reports
- `xplainer.py`: experimental script used for including explicability to the trained deep learning models
- `prz`: main library
- `scripts`: support scripts for developing the project
- `tools`: useful and external source code, such as: texture descriptors and adapted LibSVM `easy.py` file
- `tools/outdated`: scripts developed in the first iteration of this project. May be used for understanding the development flow

## Citation
Whenever using the here available code, remember to cite the original paper:
```
@article{felipe_cancer_2022,
	title = {Cancer Identification in Enteric Nervous System Preclinical Images Using Handcrafted and Automatic Learned Features},
	author = {
      Felipe, Gustavo Z.
        and Teixeira, Lucas O.
        and Pereira, Rodolfo M.
        and Zanoni, Jacqueline N.
        and Souza, Sara R. G.
        and Nanni, Loris
        and Cavalcanti, George D. C.
        and Costa, Yandre M. G.},
	issn = {1573-773X},
	url = {https://doi.org/10.1007/s11063-022-11114-y},
	doi = {10.1007/s11063-022-11114-y},
	journal = {Neural Processing Letters},
	month = dec,
	year = {2022},
}
```

## Additional Notes
1. The dataset used in this project is publically available at: https://github.com/gustavozf/ENS_dataset
2. Please check out our [previous dataset (EGC-Z)](https://github.com/gustavozf/EGC_Z_dataset) if you are interest into developing projects with Enteric Nervous System images

## License 
This work is licensed under the Apache 2.0 License. Please check `LICENSE` file for more details.