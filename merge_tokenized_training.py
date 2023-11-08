"""
Merges two tokenized dataset files into one training set.
Tokens in the second tokenizer that are equal to some token in the first tokenizer
are set to the first tokenizer token_id. The remaining tokens in the second
tokenizer are mapped to new token ids.
Note that the output training dataset is not shuffled.

Sample usage:

python3 curse-of-multilinguality/merge_tokenized_training.py \
--tokenized1="datasets_tokenized_split/eng_latn_train8k.txt" \
--tokenized2="multilingual_data_tokenized/multilingual_similar/eng_latn_10similar_medhigh_8k.txt" \
--tokenizer1="tokenizers/monolingual/eng_latn_10k" \
--tokenizer2="tokenizers/multilingual_similar/eng_latn_10similar_medhigh" \
--epochs1=20 \
--epochs2=1 \
--output_path="temp/eng_latn_8k_10similar_medhigh_8k.txt" \
--vocab_size_outpath="temp/eng_latn_10similar_medhigh_vocabsize.txt"

"""

import os
import codecs
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.data_utils import MergedTokenizer

# Output vocab size will be padded to a multiple of this.
VOCAB_MULTIPLE = 2048

def create_parser():
    parser = argparse.ArgumentParser()
    # Output tokenized training data.
    parser.add_argument('--output_path', type=str, required=True)
    # Output vocab size to a text file. The only contents are "vocab_size".
    parser.add_argument('--vocab_size_outpath', type=str, required=True)
    # Tokenized data files.
    parser.add_argument('--tokenized1', type=str, required=True)
    parser.add_argument('--tokenized2', type=str, required=True)
    # Tokenizer paths.
    parser.add_argument('--tokenizer1', type=str, required=True)
    parser.add_argument('--tokenizer2', type=str, required=True)
    # Epochs through the data.
    parser.add_argument('--epochs1', type=int, required=True)
    parser.add_argument('--epochs2', type=int, required=True)
    return parser


def main(args):
    tokenizer1 = AutoTokenizer.from_pretrained(args.tokenizer1)
    tokenizer2 = AutoTokenizer.from_pretrained(args.tokenizer2)
    merge_tokenizer = MergedTokenizer(tokenizer1, tokenizer2, str_ids=True)
    outfile = codecs.open(args.output_path, 'wb', encoding='utf-8')
    print('Copying {} epochs of dataset 1.'.format(args.epochs1))
    for epoch_i in range(args.epochs1):
        # Copy the training data. This tokenization is unaffected.
        infile = codecs.open(args.tokenized1, 'rb', encoding='utf-8')
        for line in tqdm(infile):
            outfile.write(line)
        infile.close()
    print('Adding {} epochs of dataset 2.'.format(args.epochs2))
    for epoch_i in range(args.epochs2):
        # Add the second training set, with the merged tokenization.
        infile = codecs.open(args.tokenized2, 'rb', encoding='utf-8')
        for line in tqdm(infile):
            example = line.strip().split()
            updated_example = merge_tokenizer.get_merged_ids(example, tokenizer1=False)
            outfile.write(' '.join(updated_example))
            outfile.write('\n')
        infile.close()
    outfile.close()

    # Output the vocab size with some added tokens for efficiency.
    vocab_size = merge_tokenizer.vocab_size
    if vocab_size % VOCAB_MULTIPLE != 0:
        vocab_size = VOCAB_MULTIPLE * ((vocab_size // VOCAB_MULTIPLE) + 1)
    outfile = codecs.open(args.vocab_size_outpath, 'wb', encoding='utf-8')
    outfile.write(str(vocab_size))
    outfile.close()
    print('Merged vocab size: {}'.format(vocab_size))
    print('Done')


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
