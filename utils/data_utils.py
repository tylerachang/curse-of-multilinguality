"""
Utilities for text data and tokenization.
"""

import os
import codecs
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm


# To process lines in batches for tokenization.
MAX_STORED_LINE_COUNT = 10000

# Returns the vocabulary given a path to a tokenizer.
# Returns as a set or as a list.
def get_vocab(tokenizer_path, first_n=None, as_set=True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if first_n is None:
        vocab = tokenizer.get_vocab().keys()
        # Exclude special tokens of this form.
        vocab = [w for w in vocab if not w.startswith('[XXXXX')]
    else:
        # Excludes special tokens.
        vocab = tokenizer.convert_ids_to_tokens(np.arange(tokenizer.vocab_size))
        vocab = vocab[:first_n]
    if as_set:
       return set(vocab)
    else:
        return vocab


# Return a random subset of the lines in a text file.
# Returns a list of strings.
# Note: lines still have \n at the end.
def get_lines_subset(inpath, n_lines, total_lines=None):
    total_examples = total_lines
    if total_examples is None:
        # Count the total lines.
        # Assume no empty lines, faster.
        total_examples = 0
        infile = codecs.open(inpath, 'rb', encoding='utf-8')
        for line in tqdm(infile):
            total_examples += 1
        infile.close()
    # Create a mask.
    if n_lines > total_examples:
        print('WARNING: trying to select {0} lines from {1} lines.'.format(n_lines, total_examples))
        n_lines = total_examples
    mask = np.zeros(total_examples, dtype=bool)
    to_keep = np.random.choice(total_examples, size=n_lines, replace=False)
    mask[to_keep] = True
    # Select the lines.
    lines_subset = []
    infile = codecs.open(inpath, 'rb', encoding='utf-8')
    for line_i, line in tqdm(enumerate(infile)):
        if mask[line_i]:
            lines_subset.append(line)
    infile.close()
    return lines_subset


# Writes a random subset of the lines in a text file.
# Can write (w) or append (a).
def write_lines_subset(inpath, n_lines, outpath, outmode='w', total_lines=None):
    total_examples = total_lines
    if total_examples is None:
        # Count the total lines.
        # Assume no empty lines, faster.
        total_examples = 0
        infile = codecs.open(inpath, 'rb', encoding='utf-8')
        for line in tqdm(infile):
            total_examples += 1
        infile.close()
    # Create a mask.
    if n_lines > total_examples:
        print('WARNING: trying to select {0} lines from {1} lines.'.format(n_lines, total_examples))
        n_lines = total_examples
    mask = np.zeros(total_examples, dtype=bool)
    to_keep = np.random.choice(total_examples, size=n_lines, replace=False)
    mask[to_keep] = True
    # Select the lines.
    outfile = codecs.open(outpath, outmode + 'b', encoding='utf-8')
    infile = codecs.open(inpath, 'rb', encoding='utf-8')
    for line_i, line in tqdm(enumerate(infile)):
        if mask[line_i]:
            outfile.write(line)
    infile.close()
    outfile.close()
    return True


# Tokenize a file.
# Each example concatenates a maximum of max_segments lines, up to max_seq_len
# tokens. Produces up to max_examples examples.
def tokenize_file(input_path, output_path, tokenizer,
                  max_examples, max_segments, max_seq_len,
                  add_special_tokens=True, overwrite=False):
    print("Tokenizing file: {}".format(input_path))
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    if add_special_tokens and (cls_token_id is None or sep_token_id is None):
        print("Warning: [CLS] or [SEP] token does not exist.")

    if os.path.isfile(output_path) and not overwrite:
        print("File already exists: {}".format(output_path))
        return -1
    infile = codecs.open(input_path, 'rb', encoding='utf-8')
    outfile = codecs.open(output_path, 'wb', encoding='utf-8')
    example_count = 0
    stored_lines = []

    # Process and reset the stored lines.
    def process_stored_lines():
        nonlocal stored_lines
        nonlocal example_count
        batch_encoding = tokenizer(stored_lines, add_special_tokens=False, truncation=True,
                                   max_length=max_seq_len, return_attention_mask=False,
                                   return_token_type_ids=False)
        curr_example = [cls_token_id] if add_special_tokens else []
        curr_n_segments = 0
        for tokenized_line in batch_encoding["input_ids"]:
            curr_example = curr_example + tokenized_line
            if add_special_tokens: curr_example = curr_example + [sep_token_id]
            curr_n_segments += 1
            if len(curr_example) >= max_seq_len or curr_n_segments >= max_segments:
                # Process an example.
                curr_example = curr_example[:max_seq_len]
                # Note that these examples are unpadded.
                outfile.write(" ".join(str(token_id) for token_id in curr_example))
                outfile.write('\n')
                curr_example = [cls_token_id] if add_special_tokens else []
                curr_n_segments = 0
                example_count += 1
                if example_count >= max_examples:
                    return
        stored_lines = []
        return

    # Process infile.
    for line_i, line in enumerate(infile):
        stripped_line = line.strip()
        if stripped_line != '':
            stored_lines.append(stripped_line)
        # Process the currently stored lines.
        if line_i % MAX_STORED_LINE_COUNT == MAX_STORED_LINE_COUNT-1:
            process_stored_lines()
            if example_count >= max_examples:
                break
            print("Processed up to line {0} ({1} examples)".format(line_i, example_count))
    infile.close()
    # Process the remaining set of lines. This is copied from above for maximal bad code style!
    if example_count < max_examples and len(stored_lines) > 0:
        process_stored_lines()
    outfile.close()
    print("Finished tokenization: {} examples.".format(example_count))
    return example_count


# Converts tokenized sequences into a common form for two tokenizers.
# Tokens in the second tokenizer that are equal to some token in the first tokenizer
# are set to the first tokenizer token_id. The remaining tokens in the second
# tokenizer are mapped to new token ids.
class MergedTokenizer:
    def __init__(self, tokenizer1, tokenizer2, str_ids=False):
        print('Merging tokenizers.')
        self.tokenizer2_map = dict()
        # Vocab lists.
        vocab1 = tokenizer1.convert_ids_to_tokens(np.arange(tokenizer1.vocab_size))
        vocab2 = tokenizer2.convert_ids_to_tokens(np.arange(tokenizer2.vocab_size))
        # Special tokens.
        if str_ids: # Token ids are strings (e.g. "1").
            self.tokenizer2_map[str(tokenizer2.cls_token_id)] = str(tokenizer1.cls_token_id)
            self.tokenizer2_map[str(tokenizer2.sep_token_id)] = str(tokenizer1.sep_token_id)
            self.tokenizer2_map[str(tokenizer2.pad_token_id)] = str(tokenizer1.pad_token_id)
            self.tokenizer2_map[str(tokenizer2.mask_token_id)] = str(tokenizer1.mask_token_id)
        else: # Token ids are ints (e.g. 1).
            self.tokenizer2_map[tokenizer2.cls_token_id] = tokenizer1.cls_token_id
            self.tokenizer2_map[tokenizer2.sep_token_id] = tokenizer1.sep_token_id
            self.tokenizer2_map[tokenizer2.pad_token_id] = tokenizer1.pad_token_id
            self.tokenizer2_map[tokenizer2.mask_token_id] = tokenizer1.mask_token_id
        # Assume only four special tokens.
        # I.e. ignore tokens that were added to make len(tokenizer) a multiple of a 2^n.
        next_available = tokenizer1.vocab_size + 4
        for tokenizer2_id, token in enumerate(vocab2):
            if token in vocab1:
                new_id = vocab1.index(token)
            else:
                new_id = next_available
                next_available += 1
            # Update the tokenization map.
            if str_ids:
                self.tokenizer2_map[str(tokenizer2_id)] = str(new_id)
            else:
                self.tokenizer2_map[tokenizer2_id] = new_id
        # Including the four special tokens.
        self.vocab_size = next_available
        print('New vocab size: {}'.format(self.vocab_size))

    # Converts a tokenized sequence (list of integers) into the merged token ids.
    # The input tokenized sequence is either from tokenizer1 or tokenizer2.
    def get_merged_ids(self, sequence, tokenizer1=True):
        if tokenizer1:
            # Tokens from tokenizer1 are unchanged.
            return sequence
        else:
            return [self.tokenizer2_map[tokenizer2_id] for tokenizer2_id in sequence]
