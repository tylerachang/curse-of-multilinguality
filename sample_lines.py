"""
Sample 10K lines from all languages, to train multilingual tokenizers.
Hard-coded.

Sample usage:

python3 curse-of-multilinguality/sample_lines.py

"""

import os
from utils.data_utils import write_lines_subset
from utils.constants import LANG_SETS

INPUT_DIR = 'datasets'
OUTPUT_DIR = 'datasets_10k'
N_LINES = 10000

def main():
    for lang in LANG_SETS['low']:
        inpath = os.path.join(INPUT_DIR, '{}.txt'.format(lang))
        outpath = os.path.join(OUTPUT_DIR, '{}.txt'.format(lang))
        if os.path.isfile(outpath):
            print('Already found file, skipping: {}'.format(outpath))
            continue
        write_lines_subset(inpath, N_LINES, outpath, outmode='w')
    print('Done')


if __name__ == "__main__":
    main()
