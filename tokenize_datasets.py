"""
Trains tokenizers and tokenizes all the files.

Sample usage:

python3 curse-of-multilinguality/tokenize_datasets.py

"""

import os
import codecs
import sentencepiece as spm

VOCAB_SIZE = 32000
INPUT_DIR = 'datasets'
OUTPUT_DIR = 'datasets_tokenized'
TOKENIZERS_DIR = 'tokenizers'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(TOKENIZERS_DIR, 'spm'), exist_ok=True)
os.makedirs(os.path.join(TOKENIZERS_DIR, 'monolingual'), exist_ok=True)
monolingual
fnames = os.listdir(INPUT_DIR)
langs = [fname.split('.')[0] for fname in fnames]
for lang in sorted(langs):
    outpath = os.path.join(OUTPUT_DIR, '{}_tokenized.txt'.format(lang))
    if os.path.isfile(outpath):
        print('Already found file: {}'.format(outpath))
        continue
    print('Tokenizing lang: {}'.format(lang))
    inpath = os.path.join(INPUT_DIR, '{}.txt'.format(lang))
    spm_tokenizer_path = os.path.join(TOKENIZERS_DIR, 'spm/{}_10k'.format(lang))
    try:
        spm.SentencePieceTrainer.train(input=inpath,
                model_prefix=spm_tokenizer_path,
                vocab_size=VOCAB_SIZE,
                input_sentence_size=10000,
                train_extremely_large_corpus=True,
                shuffle_input_sentence=True,
                num_threads=16)
    except RuntimeError as e:
        error_message = str(e)
        if '!sentences_.empty()' in error_message:
            print('Empty input file: {}'.format(inpath))
            # Write empty output file.
            outfile = codecs.open(outpath, 'wb', encoding='utf-8')
            outfile.close()
            continue
        assert 'Vocabulary size too high' in error_message
        # Max vocab size is the last word in the message.
        new_vocab_size = int(error_message.split()[-1].replace('.', ''))
        print('Changing vocab size to {}.'.format(new_vocab_size))
        spm.SentencePieceTrainer.train(input=inpath,
                model_prefix=spm_tokenizer_path,
                vocab_size=new_vocab_size,
                input_sentence_size=10000,
                train_extremely_large_corpus=True,
                shuffle_input_sentence=True,
                num_threads=16)
    hf_tokenizer_path = os.path.join(TOKENIZERS_DIR, 'monolingual/{}_10k'.format(lang))
    command = """python3 ../word-acquisition-language-models/scripts/convert_spm_to_hf_tokenizer.py \
    --input={0}.model \
    --output_dir={1} \
    --keep_accents=True \
    --multiple_of=2048""".format(spm_tokenizer_path, hf_tokenizer_path)
    result = os.popen(command).read()
    print(result)
    command = """python3 ../word-acquisition-language-models/scripts/tokenize_dataset.py \
    --tokenizer={0} \
    --input_file={1} \
    --output_file={2} \
    --max_segments=-1 --max_seq_len=128 --max_examples=10000000000""".format(hf_tokenizer_path, inpath, outpath)
    result = os.popen(command).read()
    print(result)
    print('Finished lang: {}'.format(lang))
