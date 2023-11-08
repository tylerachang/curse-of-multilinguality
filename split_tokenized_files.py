"""
Create train and eval splits for monolingual models. Assumes shuffled tokenized
input datasets. The last 4K examples are used as eval sets. Then, 8*10^k
remaining examples are sampled for training (for maximal k). Then, the smaller
training sets (lower k) are iteratively sampled from the training set. This
creates low-resource, ..., high-resource training sets with 8K, 80K, 800K, and
8M examples.

Assumes the input filenames are [LANG]_tokenized.txt.

Sample usage:

python3 curse-of-multilinguality/split_tokenized_files.py \
--input_dir="datasets_tokenized_8004k" \
--output_dir="datasets_tokenized_split"

"""

import os
import codecs
import argparse
import numpy as np
from tqdm import tqdm


EVAL_SIZE = 4000
# Should be sorted.
TRAIN_SIZES = [8000, 80000, 800000, 8000000]
TRAIN_NAMES = ['train8k', 'train80k', 'train800k', 'train8000k']


def create_parser():
    parser = argparse.ArgumentParser()
    # Directory containing shuffled tokenized datasets.
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser


def main(args):
    # Create output directory.
    os.makedirs(args.output_dir, exist_ok=True)
    # Loop through the files.
    for fname in os.listdir(args.input_dir):
        inpath = os.path.join(args.input_dir, fname)
        if not os.path.isfile(inpath):
            continue
        # lang = fname.split('_')[0]
        lang = fname.replace('_tokenized.txt', '')
        infile = codecs.open(inpath, 'rb', encoding='utf-8')
        # Assume no empty lines.
        print('Counting lines in {}'.format(fname))
        total_examples = 0
        for line in tqdm(infile):
            total_examples += 1
        infile.close()
        # Create masks for each split.
        print('Creating split masks.')
        data_masks = dict()
        # The last EVAL_SIZE examples are used for eval.
        if total_examples >= EVAL_SIZE:
            data_masks['eval'] = np.zeros(total_examples, dtype=bool)
            data_masks['eval'][-EVAL_SIZE:] = True
        # Get train splits.
        train_splits = list(zip(TRAIN_NAMES, TRAIN_SIZES))
        first_split = True
        for train_name, train_size in reversed(train_splits):
            if total_examples < train_size + EVAL_SIZE:
                continue
            if first_split:
                # Sample from the remaining (non-eval) examples.
                data_mask = np.zeros(total_examples, dtype=bool)
                to_keep = np.random.choice(total_examples-EVAL_SIZE, size=train_size, replace=False)
                data_mask[to_keep] = True
                data_masks[train_name] = data_mask
                first_split = False
            else:
                # Sample from the previous split.
                data_mask = np.zeros(total_examples, dtype=bool)
                to_keep = np.random.choice(to_keep, size=train_size, replace=False)
                data_mask[to_keep] = True
                data_masks[train_name] = data_mask
        if len(data_masks) < 2:
            # No training splits.
            print('Not enough examples in {}'.format(fname))
            continue
        # Create splits.
        print('Creating splits.')
        infile = codecs.open(inpath, 'rb', encoding='utf-8')
        outfiles = dict()
        to_skip = []
        for split_name in data_masks:
            outpath = os.path.join(args.output_dir, '{0}_{1}.txt'.format(lang, split_name))
            if os.path.isfile(outpath):
                print('WARNING: file already exists: {}'.format(outpath))
                to_skip.append(split_name)
                continue
            outfiles[split_name] = codecs.open(outpath, 'wb', encoding='utf-8')
        # Skip splits that already exist.
        for split_name in to_skip:
            data_masks.pop(split_name)
        if len(data_masks) == 0:
            continue
        print('Creating splits: {}'.format(list(data_masks.keys())))
        for line_i, line in tqdm(enumerate(infile)):
            for split_name, data_mask in data_masks.items():
                # Check if this split contains this example.
                if data_mask[line_i]:
                    outfiles[split_name].write(line)
        infile.close()
        for split_name, outfile in outfiles.items():
            outfile.close()
        print('Finished {}'.format(fname))
    print('Done')


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
