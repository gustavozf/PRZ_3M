import os
import argparse

import numpy as np
import pandas as pd

import prz.analysis.classification as clf_an
from prz.utils.io import dump_json
from prz.utils.plot import plot_acc_comparison

ANIM_TAGS = [
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
]

def get_group_ids(input_path: str):
    group_data = pd.read_csv(input_path)
    anim_tags =  group_data['anim_tag'].unique()
    filter = lambda x : group_data.anim_tag == x

    return {
        anim_tag: (
            group_data[filter(anim_tag)]['anim_tag_id'].unique()[0]
        )
        for anim_tag in anim_tags
    }

def get_args():
    parser = argparse.ArgumentParser(
        description='Generate classification stats from the preditions.'
    )

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=True,
        nargs='+',
        help='Input CSV file containing the predictions.'
    )
    parser.add_argument(
        '-g',
        '--groups',
        type=str,
        required=True,
        nargs='+',
        help='CSV file containing the groups IDs.'
    )
    parser.add_argument(
        '-t',
        '--tags',
        type=str,
        required=True,
        nargs='+',
        help='Experiment tags.'
    )

    return parser.parse_args()

def main():
    args = get_args()

    assert len(args.input) == len(args.groups) == len(args.tags)

    plotable_accs = {}

    for zipped_args in zip(args.input, args.groups, args.tags):
        input_path, groups_path, tag = zipped_args
        print('Processing: ', input_path)

        data = pd.read_csv(input_path)
        groups = get_group_ids(groups_path)
        accs = {
            _group : clf_an.accuracy_from_labels(
                data[data.group_id == _group_id]['y_true'],
                data[data.group_id == _group_id]['y_pred'],
            )
            for _group, _group_id in groups.items()
        }
        plotable_accs[tag] = [accs[item] for item in ANIM_TAGS]

        dump_json(
            {
                'accs': accs,
                **clf_an.get_stats_report(list(accs.values()))
            },
            os.path.join(os.path.dirname(input_path), 'acc_stats.json')
        )

    plot_acc_comparison(plotable_accs, './', ANIM_TAGS)

if __name__ == '__main__':
    main()