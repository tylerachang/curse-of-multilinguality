"""
Prepares for multilingual pretraining.
For each target language, selects languages to add, creates a tokenizer for the
added languages, tokenizes the added language datasets, and samples the correct
amount in each added language.
Outputs:
Bash scripts to run language model pretraining for all target languages.
The file of tokenized examples to add to pretraining for each target language.
The tokenizers for the added languages for each target language.
A log of the added languages for each target language.

Notes on temp files:
* When tokenizing added languages with the multilingual tokenizers, the full
tokenized dataset is saved as a temp file associated with the unique script id
(to allow multiple instances of the script to run simultaneously). This dataset
is sampled to select the added data in that language.
* The SentencePiece tokenizer for added languages is saved in the temp dir, also
associated with the script id, before conversion to a Hugging Face tokenizer.
The sampled training data for the tokenizer is also saved in the temp dir,
associated with the script id.
* Because the merged training data for each model (the original monolingual data
epochs plus the added language data) is large, this is compiled on the fly
when the bash scripts are run. These merged training datasets are saved in the
temp dir, associated with a script id assigned to each individual output bash
script. This allows the different bash scripts to be run simultaneously.

Note: lang2vec is not available through conda, so need to run in a separate
environment:
python3 -m pip install lang2vec
Then, run get_similar_languages() once, which will cache the distances. Then,
the scripts can run in other conda environments after.
cd curse-of-multilinguality
python3
from utils.distance_utils import get_lang_similarities
from utils.constants import LANGS_100K
get_lang_similarities(LANGS_100K, '../lang_similarities_dir', '../all_monolingual_tokenizers/[lang]_10k')
cd ..

Sample usage:

python3 curse-of-multilinguality/prepare_multilingual_training.py \
--outdir="multilingual_data_prep/multilingual_similar" \
--out_data_dir="multilingual_data_tokenized/multilingual_similar" \
--out_models_dir="models/multilingual_similar" \
--out_tokenizers_dir="tokenizers/multilingual_similar" \
--added_langs_selection="similar" --added_langs_n=10 \
--added_langs_superset="medhigh" \
--data_text_dir="datasets" --data_tokenizer_text_dir="datasets_10k" \
--data_monolingual_tokenized_dir="datasets_tokenized_split" \
--in_tokenizers_dir="tokenizers/monolingual" \
--lang_similarities_dir="lang_similarities_dir"

Then run the bash script for multilingual training:

chmod u+x multilingual_data_prep/multilingual_similar/*.sh
multilingual_data_prep/multilingual_similar/train_low_10similar_medhigh_8k_small.sh

"""

import os
import codecs
import argparse
import numpy as np
import math

from utils.constants import DATA_QUANTITIES, DATA_QUANTITIES_STR, LANG_SETS, MODEL_SIZES, oscar103_code_mapping, LANGS_100K
from utils.distance_utils import get_similar_languages
from utils.data_utils import write_lines_subset, tokenize_file

import sentencepiece as spm
from transformers import AlbertTokenizer, AutoTokenizer

# Fixed defaults.
# The total vocab size for the added languages.
OUTLANG_VOCAB_SIZE = 32000
# Tokenizer training lines per added language.
TOKENIZER_TRAIN_PER_LANG = 10000
# For added languages, the number of examples that are tokenized.
# Added data is sampled from these examples.
MAX_TOKENIZATION = 8000000
MAX_SEQ_LENGTH = 128
TOTAL_TRAIN_BATCH_SIZE = 128
WARMUP_PROPORTION = 0.10
EVAL_STEPS = 1000
MONOLINGUAL_EPOCHS = {'low': 20, 'medlow': 20,
                      'medhigh': 10, 'high': 2}


def create_parser():
    parser = argparse.ArgumentParser()
    # Where to output:
    # log, including arguments and the languages added for each target language.
    # bash scripts to run pretraining.
    parser.add_argument('--outdir', type=str, required=True)
    # Temporary files: tokenizer training data, SPM tokenizer before conversion
    # to HF, full tokenized data for each added language (which is sampled),
    # Merged data during training.
    parser.add_argument('--temp_dir', type=str, default='temp')

    # The raw text data in each language.
    parser.add_argument('--data_text_dir', type=str, required=True)
    # Raw text, but a smaller quantity, for training the tokenizer.
    # This can be the same as data_text_dir, but will be slower.
    parser.add_argument('--data_tokenizer_text_dir', type=str, required=True)
    # The data tokenized with the monolingual tokenizers.
    parser.add_argument('--data_monolingual_tokenized_dir', type=str, required=True)
    parser.add_argument('--out_models_dir', type=str, required=True)
    # To output the tokenized data to add for each language.
    parser.add_argument('--out_data_dir', type=str, required=True)
    parser.add_argument('--out_tokenizers_dir', type=str, required=True)

    # The monolingual tokenizers.
    parser.add_argument('--in_tokenizers_dir', type=str, required=True)
    parser.add_argument('--hf_cache', type=str, default='hf_cache')

    # Similarities between languages.
    # Contains the languages (lang_key.txt) and different similarity files:
    # syntax.npy, phonology.npy, inventory.npy, geo.npy, vocab.npy
    parser.add_argument('--lang_similarities_dir', type=str, default='lang_similarities')

    # Currently supported: similar, dissimilar.
    parser.add_argument('--added_langs_selection', type=str, default='similar')
    # How many languages to add.
    parser.add_argument('--added_langs_n', type=int, default=10)
    # Select languages to add from this set.
    # This also defines how many sequences to add per language.
    # Saves the tokenized data that can be added for other amounts too, up to this amount.
    parser.add_argument('--added_langs_superset', type=str, default='medhigh')
    return parser


# For a given target language:
# Selects languages to add, creates a tokenizer for the added languages,
# tokenizes the added language datasets, and samples the correct
# amount in each added language.
def prepare_language(target_lang, args, script_id):
    # Check if the path already exists for the old language code.
    # These are identical to below other than the changed language code.
    if target_lang in oscar103_code_mapping:
        old_code = oscar103_code_mapping[target_lang]
        tokenizer_name = '{0}_{1}{2}_{3}'.format(old_code, args.added_langs_n,
                args.added_langs_selection, args.added_langs_superset)
        tokenizer_path = os.path.join(args.out_tokenizers_dir, tokenizer_name)
        data_quant_str = DATA_QUANTITIES_STR[args.added_langs_superset]
        data_outname = '{0}_{1}{2}_{3}_{4}.txt'.format(old_code, args.added_langs_n,
                args.added_langs_selection, args.added_langs_superset, data_quant_str)
        data_outpath = os.path.join(args.out_data_dir, data_outname)
        if os.path.isdir(tokenizer_path) and os.path.isfile(data_outpath):
            print('Already found tokenizer and data for old language code: {0} ({1})'.format(old_code, target_lang))
            return

    # Check if tokenizer and tokenized data to add already exist.
    tokenizer_name = '{0}_{1}{2}_{3}'.format(target_lang, args.added_langs_n,
            args.added_langs_selection, args.added_langs_superset)
    tokenizer_path = os.path.join(args.out_tokenizers_dir, tokenizer_name)
    # Check for largest amount of added data.
    data_quant_str = DATA_QUANTITIES_STR[args.added_langs_superset]
    data_outname = '{0}_{1}{2}_{3}_{4}.txt'.format(target_lang, args.added_langs_n,
            args.added_langs_selection, args.added_langs_superset, data_quant_str)
    data_outpath = os.path.join(args.out_data_dir, data_outname)
    if os.path.isdir(tokenizer_path) and os.path.isfile(data_outpath):
        print('Already found tokenizer and data for target language: {}'.format(target_lang))
        return


    print('Selecting added languages for language: {}'.format(target_lang))
    # Vocabulary overlap is computed using the language 10k tokenizers.
    monolingual_tokenizer_template = os.path.join(args.in_tokenizers_dir, '[lang]_10k')
    added_langs_superset = LANG_SETS[args.added_langs_superset]
    if args.added_langs_selection == 'similar':
        added_langs = get_similar_languages(target_lang, added_langs_superset, args.added_langs_n,
                    args.lang_similarities_dir, reverse=False,
                    tokenizer_template=monolingual_tokenizer_template,
                    all_langs=LANGS_100K)
    elif args.added_langs_selection == 'dissimilar':
        added_langs = get_similar_languages(target_lang, added_langs_superset, args.added_langs_n,
                    args.lang_similarities_dir, reverse=True,
                    tokenizer_template=monolingual_tokenizer_template,
                    all_langs=LANGS_100K)
    print('Selected langs: {}'.format(added_langs))
    # Log selected languages.
    logname = '{0}{1}_{2}_log.txt'.format(args.added_langs_n,
            args.added_langs_selection, args.added_langs_superset)
    logpath = os.path.join(args.outdir, logname)
    outfile = codecs.open(logpath, 'ab', encoding='utf-8')
    outfile.write('Target lang: {}\n'.format(target_lang))
    outfile.write('Added langs: {}\n\n'.format(added_langs))
    outfile.close()

    # Compile TOKENIZER_TRAIN_PER_LANG from each added language.
    os.makedirs(args.temp_dir, exist_ok=True)
    temp_tokenizer_train_path = os.path.join(args.temp_dir, 'temp_tokenizer_train_{}.txt'.format(script_id))
    print('Compiling tokenizer training data: {}'.format(temp_tokenizer_train_path))
    if os.path.isfile(temp_tokenizer_train_path):
        os.remove(temp_tokenizer_train_path)
    for added_lang in added_langs:
        inpath = os.path.join(args.data_tokenizer_text_dir, '{}.txt'.format(added_lang))
        write_lines_subset(inpath, TOKENIZER_TRAIN_PER_LANG, temp_tokenizer_train_path, outmode='a')

    # Train tokenizer for added languages.
    print('Training tokenizer.')
    tokenizer_path_spm = os.path.join(args.temp_dir, 'temp_tokenizer_spm_{}'.format(script_id))
    spm.SentencePieceTrainer.train(input=temp_tokenizer_train_path,
            model_prefix=tokenizer_path_spm, vocab_size=OUTLANG_VOCAB_SIZE,
            input_sentence_size=TOKENIZER_TRAIN_PER_LANG * args.added_langs_n,
            train_extremely_large_corpus=True,
            shuffle_input_sentence=True,
            num_threads=16)
    # Convert to HF tokenizer and save.
    tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path_spm + '.model',
            do_lower_case=False, keep_accents=True, cache_dir=args.hf_cache)
    os.makedirs(args.out_tokenizers_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    del tokenizer
    os.remove(tokenizer_path_spm + '.model')
    os.remove(tokenizer_path_spm + '.vocab')
    os.remove(temp_tokenizer_train_path)

    # Tokenize sequences in each added language.
    # Reload tokenizer, automatically attempts to use the fast version.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=args.hf_cache)
    n_tokenized_sequences = dict()
    for added_lang in added_langs:
        print('Tokenizing lang: {}'.format(added_lang))
        inpath = os.path.join(args.data_text_dir, '{}.txt'.format(added_lang))
        outpath = os.path.join(args.temp_dir, '{0}_tokenized_{1}.txt'.format(added_lang, script_id))
        example_count = tokenize_file(inpath, outpath, tokenizer,
                MAX_TOKENIZATION, math.inf, MAX_SEQ_LENGTH,
                add_special_tokens=True, overwrite=True)
        n_tokenized_sequences[added_lang] = example_count

    # Sample sequences in each added language, outputting to a file.
    # Sample for the desired data quantity, and all less.
    os.makedirs(args.out_data_dir, exist_ok=True)
    data_quant_i = list(DATA_QUANTITIES.keys()).index(args.added_langs_superset)
    data_quants = list(DATA_QUANTITIES.keys())[:data_quant_i+1]
    for added_data_quant in data_quants:
        # Sample for this amount of data per added language.
        # Note: this will be unshuffled across languages. The bash script
        # will shuffle this data.
        n_lines = DATA_QUANTITIES[added_data_quant]
        data_quant_str = DATA_QUANTITIES_STR[added_data_quant]
        print('Sampling {} examples for each lang.'.format(data_quant_str))
        outname = '{0}_{1}{2}_{3}_{4}.txt'.format(target_lang, args.added_langs_n,
                args.added_langs_selection, args.added_langs_superset, data_quant_str)
        outpath = os.path.join(args.out_data_dir, outname)
        for added_lang in added_langs:
            inpath = os.path.join(args.temp_dir, '{0}_tokenized_{1}.txt'.format(added_lang, script_id))
            write_lines_subset(inpath, n_lines, outpath, outmode='a', total_lines=n_tokenized_sequences[added_lang])
    # Remove the temporary files.
    for added_lang in added_langs:
        path = os.path.join(args.temp_dir, '{0}_tokenized_{1}.txt'.format(added_lang, script_id))
        os.remove(path)
    return


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    logname = '{0}{1}_{2}_log.txt'.format(args.added_langs_n,
            args.added_langs_selection, args.added_langs_superset)
    logpath = os.path.join(args.outdir, logname)
    outfile = codecs.open(logpath, 'wb', encoding='utf-8')
    outfile.write('Args:\n{}\n\n'.format(vars(args)))
    outfile.close()

    # Used for temp files.
    script_id = str(np.random.randint(1000000, high=9999999))

    # Prepare the data.
    # Run preparation for all languages.
    for target_lang in LANG_SETS['low']:
        prepare_language(target_lang, args, script_id)
    print('Done preparing data.')

    # Write the bash scripts.
    # For example:
    # 3 quantities of added data (low, medlow, medhigh)
    # 5 model sizes (tiny, mini, small, medium, base)
    # 4 target language sets (low, medlow, medhigh, high)
    # Skip:
    # low resource added unless low or medlow resource target (because the low
    # resource added is too small otherwise).
    #
    # Then: 3*5*4 - 1*5*2 = 50.
    print('Writing bash scripts.')
    os.makedirs(args.out_models_dir, exist_ok=True)
    data_quant_i = list(DATA_QUANTITIES.keys()).index(args.added_langs_superset)
    data_quants = list(DATA_QUANTITIES.keys())[:data_quant_i+1]
    script_i = 0 # Each bash script has its own script_id, to prevent overwriting when multiple are run simultaneously.
    for added_data_quant in data_quants: # How much added per language (low, medlow, medhigh, or high).
        for model_size in MODEL_SIZES: # tiny, mini, small, medium, or base.
            os.makedirs(os.path.join(args.out_models_dir, model_size), exist_ok=True)
            for lang_set in LANG_SETS.keys(): # low, medlow, medhigh, or high.
                if added_data_quant == 'low' and lang_set not in ['low', 'medlow']:
                    continue
                # Script to train on lang_set target languages with lang_set
                # data in the target language. With added_data_quant in each
                # added language, and model size model_size.
                monolingual_data_quant_str = DATA_QUANTITIES_STR[lang_set]
                added_data_quant_str = DATA_QUANTITIES_STR[added_data_quant]
                script_outname = 'train_{0}_{1}{2}_{3}_{4}_{5}.sh'.format(lang_set, args.added_langs_n,
                        args.added_langs_selection, args.added_langs_superset, added_data_quant_str,
                        model_size)
                script_outpath = os.path.join(args.outdir, script_outname)
                script = ''
                # First, set settings.
                # Training defaults:
                # tiny: one device, 32 batch, 4 gradient accumulation.
                # mini: one device, 32 batch, 4 gradient accumulation.
                # small: one device, 32 batch, 4 gradient accumulation.
                # medium: one device, 32 batch, 4 gradient accumulation.
                # base: four devices, 32 batch per device
                if model_size == 'base':
                    script += """
                        export MODEL_SIZE="base"
                    """
                    batch_per_device = 32
                    n_devices = 4
                    lr = 0.0001
                elif model_size == 'medium':
                    script += """
                        export MODEL_SIZE="medium"
                    """
                    batch_per_device = 32
                    n_devices = 1
                    lr = 0.0002
                elif model_size == 'small':
                    script += """
                        export MODEL_SIZE="small"
                    """
                    batch_per_device = 32
                    n_devices = 1
                    lr = 0.0005
                    if monolingual_data_quant_str == '8k': lr = 0.0002
                elif model_size == 'mini':
                    script += """
                        export MODEL_SIZE="mini"
                    """
                    batch_per_device = 32
                    n_devices = 1
                    lr = 0.0007
                    if monolingual_data_quant_str == '8k': lr = 0.0004
                elif model_size == 'tiny':
                    script += """
                        export MODEL_SIZE="tiny"
                    """
                    batch_per_device = 32
                    n_devices = 1
                    lr = 0.001
                # Learning rate.
                script += """
                    export LEARNING_RATE={}
                """.format(lr)
                # Determine the number of steps.
                # The entire training dataset is compiled into one file containing
                # multiple epochs of the monolingual data, so considered just one epoch.
                total_examples = DATA_QUANTITIES[lang_set] * MONOLINGUAL_EPOCHS[lang_set]
                total_examples += DATA_QUANTITIES[added_data_quant] * args.added_langs_n
                steps = total_examples // TOTAL_TRAIN_BATCH_SIZE
                warmup_steps = int(steps * WARMUP_PROPORTION)
                eval_steps = EVAL_STEPS
                if monolingual_data_quant_str == '8k' and added_data_quant_str == '8k':
                    eval_steps = 100  # Eval more frequently.
                if monolingual_data_quant_str == '8k' and added_data_quant_str == '80k':
                    eval_steps = 500  # Eval more frequently.
                script += """
                    export EVAL_STRATEGY="steps"
                    export EVAL_STEPS={0}
                    export MAX_STEPS={1}
                    export WARMUP_STEPS={2}
                    export N_EXAMPLES_PER_EPOCH={3}
                """.format(eval_steps, steps, warmup_steps, total_examples)
                # Training.
                langs_str = ' '.join(LANG_SETS[lang_set])
                script += """
                    for LANG in {0}; do
                """.format(langs_str)
                # Check if model already exists.
                model_outname = '${{LANG}}_{0}_{1}{2}_{3}_{4}_{5}'.format(monolingual_data_quant_str,
                        args.added_langs_n, args.added_langs_selection, args.added_langs_superset,
                        added_data_quant_str, model_size)
                model_outpath = os.path.join(args.out_models_dir, model_size, model_outname)
                model_filepath = os.path.join(model_outpath, 'pytorch_model.bin')
                script += """
                if test -f {0}; then
                echo "Model already found; skipping ${{LANG}}."
                continue
                fi
                """.format(model_filepath)
                # Merge the tokenized training data (monolingual and multilingual).
                added_data_name = '${{LANG}}_{0}{1}_{2}_{3}.txt'.format(args.added_langs_n,
                        args.added_langs_selection, args.added_langs_superset, added_data_quant_str)
                added_data_path = os.path.join(args.out_data_dir, added_data_name)
                monolingual_tokenizer_path = os.path.join(args.in_tokenizers_dir, '${LANG}_10k')
                multi_tokenizer_name = '${{LANG}}_{0}{1}_{2}'.format(args.added_langs_n,
                        args.added_langs_selection, args.added_langs_superset)
                multi_tokenizer_path = os.path.join(args.out_tokenizers_dir, multi_tokenizer_name)
                merged_outname = '${{LANG}}_{0}_{1}{2}_{3}_{4}_{5}.txt'.format(monolingual_data_quant_str,
                        args.added_langs_n, args.added_langs_selection, args.added_langs_superset,
                        added_data_quant_str, script_id+'_'+str(script_i))
                merged_outpath = os.path.join(args.temp_dir, merged_outname)
                vocab_size_outname = '${{LANG}}_{0}{1}_{2}_vocabsize_{3}.txt'.format(args.added_langs_n,
                        args.added_langs_selection, args.added_langs_superset, script_id+'_'+str(script_i))
                vocab_size_outpath = os.path.join(args.temp_dir, vocab_size_outname)
                script += """
                    python3 curse-of-multilinguality/merge_tokenized_training.py \\
                    --tokenized1="{0}/${{LANG}}_train{1}.txt" \\
                    --tokenized2={2} \\
                    --tokenizer1={3} \\
                    --tokenizer2={4} \\
                    --epochs1={5} \\
                    --epochs2=1 \\
                    --output_path={6} \\
                    --vocab_size_outpath={7}
                """.format(args.data_monolingual_tokenized_dir,
                        monolingual_data_quant_str, added_data_path,
                        monolingual_tokenizer_path, multi_tokenizer_path,
                        MONOLINGUAL_EPOCHS[lang_set], merged_outpath,
                        vocab_size_outpath)
                # Set vocab size for the merged tokenization.
                # This will be used to overwrite the tokenizer vocab size in the
                # pretraining script.
                script += """
                    export VOCAB_SIZE=$(cat {})
                """.format(vocab_size_outpath)
                # Shuffle the training data.
                # Assumes terashuf already downloaded:
                # git clone https://github.com/alexandres/terashuf.git
                # (cd terashuf && make)
                # mkdir terashuf_tmp_output
                shuffled_outpath = merged_outpath.replace('.txt', '_shuffled.txt')
                script += """
                    export TMPDIR=terashuf_tmp_output
                    export SEED=42
                    terashuf/terashuf < "{0}" > "{1}"
                    rm "{2}"
                """.format(merged_outpath, shuffled_outpath, merged_outpath)
                # Pre-training script.
                if n_devices > 1:
                    script += """
                        torchrun --nnodes=1 --nproc_per_node={} --master_port=29401 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \\
                    """.format(n_devices)
                else:
                    script += """
                        python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \\
                    """
                # Note: for training, we can use the monolingual tokenizer as a
                # placeholder, because all the datasets are iterable (i.e. already tokenized).
                # The tokenizer is only used for:
                # Resetting the vocab size in config, which we will overwrite anyways.
                # cls_token_id, sep_token_id, pad_token_id
                # For masking:
                # mask_token, and len(tokenizer) for random replacement during masking.
                # For our use cases (autoregressive models), we can use the placeholder monolingual tokenizer.
                gradient_accumulation_steps = TOTAL_TRAIN_BATCH_SIZE // (batch_per_device * n_devices)
                assert TOTAL_TRAIN_BATCH_SIZE % (batch_per_device * n_devices) == 0
                script += """
                    --tokenizer_name={0} \\
                    --config_name="gpt_{1}_config.json" \\
                    --do_train --train_iterable --eval_iterable \\
                    --eval_data_file="{2}/${{LANG}}_eval.txt" \\
                    --per_device_train_batch_size={3} --gradient_accumulation_steps={4} \\
                    --per_device_eval_batch_size=16 \\
                    --evaluation_strategy=${{EVAL_STRATEGY}} --save_strategy="no" \\
                    --eval_steps=${{EVAL_STEPS}} \\
                    --max_steps=${{MAX_STEPS}} \\
                    --warmup_steps=${{WARMUP_STEPS}} \\
                    --learning_rate=${{LEARNING_RATE}} --adam_epsilon=1e-6 --weight_decay=0.01 \\
                    --train_data_file={5} \\
                    --seed=43 --override_vocabsize=${{VOCAB_SIZE}} \\
                    --override_n_examples=${{N_EXAMPLES_PER_EPOCH}} \\
                    --output_dir={6}
                """.format(monolingual_tokenizer_path, model_size,
                        args.data_monolingual_tokenized_dir, batch_per_device,
                        gradient_accumulation_steps, shuffled_outpath, model_outpath)
                # Remove training file.
                script += """
                    rm "{0}"
                    rm "{1}"
                    done
                """.format(shuffled_outpath, vocab_size_outpath)

                # Write script to output.
                outfile = codecs.open(script_outpath, 'wb', encoding='utf-8')
                for line in script.split('\n'):
                    if line.strip() == '':
                        continue
                    outfile.write(line.strip())
                    outfile.write('\n')
                outfile.close()
                print('Wrote script: {}'.format(script_outpath))
                script_i += 1
    print('Done.')


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
