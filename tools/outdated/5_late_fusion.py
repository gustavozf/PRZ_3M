import numpy as np
import pandas as pd
import sys, os
from itertools import combinations 

# The following line is needed if the lib is being accessed localy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../PRZ_3M/')))

from prz.fusion.late_fusion import sum_rule, max_rule, product_rule
from prz.resources.data.io import DataIO

INPUT_PATH = 'D:\\Documents\\Mestrado\\Projeto\\Codes\\outputs\\'
OUTPUT_PATH = 'D:\\Documents\\Mestrado\\Projeto\\Codes\\outputs\\late_fusion\\'

IMG_TYPES = [
    'EGC', 
    'ENC', 
    'MRG',
]

ANIM_TAGS = {
    '1C',
    '2C',
    '3C',
    '4C',
    '5C',
    '6C',
    '7C',
    '22TW',
    '23TW',
    '24TW',
    '26TW',
    '28TW',
    '30TW',
}

def import_predicts(fpath, img_type):
    global ANIM_TAGS

    return {
        i : pd.read_csv(
                f'{fpath}\\{i}\\{img_type}\\{i}_{img_type}_test.csv.predict', 
                sep=' ', 
                header=0
        )
        for i in ANIM_TAGS
    }

def get_subsets(set_keys):
    subset = []
    for i in range(2, len(set_keys)+1):
        subset.extend(combinations(set_keys, i))

    return subset

def combine(pred_dict={}, rule=sum_rule):
    subsets = get_subsets(set(pred_dict.keys()))
    output_dict = {}

    # para todas as combinacoes
    for subset in subsets:
        subset_tag = ''.join(i + '_' for i in subset)
        output_dict[subset_tag] = {}
        # percorre os folds
        for tag in ANIM_TAGS:
            predicts = []
            # pega os itens das combinacoes
            print(subset, tag)
            for item in subset:
                predicts.append(pred_dict[item][tag][['0', '1']].to_numpy())

            y_pred, probas = rule(classifiers_list=predicts)
            output_dict[subset_tag][tag] = {
                'y_pred': y_pred,
                'probas': probas
            }
            
            del predicts, y_pred, probas
        del subset_tag
    del subsets

    return output_dict

def save_combinations(combinations, img_type='EGC', rule_name=''):
    global OUTPUT_PATH

    for subset in combinations.keys():
        name_tag = f'{subset}{rule_name}'

        for tag in ANIM_TAGS:
            cur_output_path = f'{OUTPUT_PATH}{img_type}\\{name_tag}\\{tag}\\'
            
            DataIO.createDirIfNotExists(cur_output_path)

            pd.DataFrame({
                'labels': combinations[subset][tag]['y_pred'],
                '0': combinations[subset][tag]['probas'][:, 0],
                '1': combinations[subset][tag]['probas'][:, 1],
            }).to_csv(
                f'{cur_output_path}{tag}_{img_type}_test.csv.predict',
                sep=' ',
                index=False,
                header=True,
            )



def main():
    global INPUT_PATH

    clf_rates = pd.read_csv(f'{INPUT_PATH}analysis\\clf_rates.csv', sep=',', header=0)

    egc_df = clf_rates[clf_rates['img_type'] == 'EGC']
    enc_df = clf_rates[clf_rates['img_type'] == 'ENC']
    mrg_df = clf_rates[clf_rates['img_type'] == 'MRG']

    top_3 = {
        'EGC': egc_df.sort_values(by=['f1_score'], ascending=False)[:3],
        'ENC': enc_df.sort_values(by=['f1_score'], ascending=False)[:3],
        'MRG': mrg_df.sort_values(by=['f1_score'], ascending=False)[:3],
    }

    for img_type in top_3.keys():
        print(img_type)
        classifiers = {}
        for tag in top_3[img_type]['extractor']:
            classifiers[tag] = import_predicts(
                f'{INPUT_PATH}\\predictions\\{tag}\\',
                img_type
            )

        for rule in [sum_rule, max_rule, product_rule]:
            combinations = combine(pred_dict=classifiers, rule=rule)
            save_combinations(combinations, img_type=img_type, rule_name=rule.__name__)


if __name__ == '__main__':
    main()