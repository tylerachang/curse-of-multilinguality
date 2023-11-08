"""
Compile eval perplexity results into a tsv.
The full_output_path includes eval loss logged during pre-training.

Sample usage:

python3 curse-of-multilinguality/compile_results.py \
--input_dir="models/monolingual/small" \
--output_path="results/monolingual_small_results.tsv"

"""

import os
import codecs
import argparse
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

FULL_RESULTS = True

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    # Will default to output_path + _full.
    parser.add_argument('--full_output_path', type=str, default='')
    return parser


def main(args):
    # Loop through the files.
    results_df = pd.DataFrame(columns=['model_name', 'ppl', 'flops'])
    full_results_df = pd.DataFrame(columns=['model_name', 'ppl', 'step'])
    for dirname in tqdm(os.listdir(args.input_dir)):
        model_name = dirname

        # Read eval_results file.
        inpath = os.path.join(args.input_dir, dirname, 'eval_results.json')
        if not os.path.isfile(inpath):
            continue
        infile = codecs.open(inpath, 'rb', encoding='utf-8')
        input = infile.read()
        infile.close()
        json_results = json.loads(input)
        final_ppl = json_results['perplexity']
        # Row dict.
        main_row_dict = {'model_name': model_name, 'ppl': final_ppl, 'flops': None}

        if not FULL_RESULTS:
            results_df = results_df.append(main_row_dict, ignore_index=True)
            continue

        # Load trainer_state file.
        inpath = os.path.join(args.input_dir, dirname, 'trainer_state.json')
        if not os.path.isfile(inpath):
            continue
        infile = codecs.open(inpath, 'rb', encoding='utf-8')
        input = infile.read()
        infile.close()
        json_results = json.loads(input)
        for log in json_results['log_history']:
            if 'eval_loss' in log:
                ppl = np.exp(log['eval_loss'])
                step = log['step']
                row_dict = {'model_name': model_name, 'ppl': ppl, 'step': step}
                full_results_df = full_results_df.append(row_dict, ignore_index=True)
        if 'max_steps' in json_results:
            max_step = int(json_results['max_steps'])
            row_dict = {'model_name': model_name, 'ppl': final_ppl, 'step': max_step}
            full_results_df = full_results_df.append(row_dict, ignore_index=True)
        if 'total_flos' in json_results:
            # Update main_row_dict with flops.
            main_row_dict['flops'] = int(json_results['total_flos'])
        # Update results.
        results_df = results_df.append(main_row_dict, ignore_index=True)

    # Save.
    results_df.to_csv(args.output_path, sep="\t", index=False)
    if args.full_output_path.strip() == '':
        full_output_path = args.output_path.replace('.tsv', '_full.tsv')
    else:
        full_output_path = args.full_output_path
    if FULL_RESULTS:
        full_results_df.to_csv(full_output_path, sep="\t", index=False)
    print('Done')


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
