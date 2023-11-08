"""
Prepares scripts to run monolingual pretraining.
Assumes the monolingual tokenizers have been trained, and the monolingual
datasets have been tokenized and split (into eval, and training sets of
different sizes).
Outputs:
Bash scripts to run language model pretraining for all languages.

Sample usage:

python3 curse-of-multilinguality/prepare_monolingual_scripts.py \
--outdir="multilingual_data_prep/monolingual" \
--out_models_dir="models/monolingual" \
--data_monolingual_tokenized_dir="datasets_tokenized_split" \
--in_tokenizers_dir="tokenizers/monolingual"

Then run the bash script for monolingual training:

chmod u+x multilingual_data_prep/monolingual/*.sh
multilingual_data_prep/monolingual/train_low_small.sh

"""

import os
import codecs
import argparse

from utils.constants import DATA_QUANTITIES, DATA_QUANTITIES_STR, LANG_SETS, MODEL_SIZES

# Fixed defaults.
TOTAL_TRAIN_BATCH_SIZE = 128
WARMUP_PROPORTION = 0.10
MONOLINGUAL_EPOCHS = {'low': 20, 'medlow': 20,
                      'medhigh': 10, 'high': 2}


def create_parser():
    parser = argparse.ArgumentParser()
    # Where to output bash scripts to run pretraining.
    parser.add_argument('--outdir', type=str, required=True)
    # The data tokenized with the monolingual tokenizers.
    parser.add_argument('--data_monolingual_tokenized_dir', type=str, required=True)
    parser.add_argument('--out_models_dir', type=str, required=True)
    # The monolingual tokenizers.
    parser.add_argument('--in_tokenizers_dir', type=str, required=True)
    parser.add_argument('--hf_cache', type=str, default='hf_cache')
    return parser


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.out_models_dir, exist_ok=True)
    for model_size in MODEL_SIZES: # tiny, mini, small, medium, or base.
        os.makedirs(os.path.join(args.out_models_dir, model_size), exist_ok=True)
        for lang_set in LANG_SETS.keys(): # low, medlow, medhigh, or high.
            # Script to train on lang_set target languages with lang_set
            # data in the target language. With model size model_size.
            monolingual_data_quant_str = DATA_QUANTITIES_STR[lang_set]
            script_outname = 'train_{0}_{1}.sh'.format(lang_set, model_size)
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
            total_examples = DATA_QUANTITIES[lang_set] * MONOLINGUAL_EPOCHS[lang_set]
            steps = total_examples // TOTAL_TRAIN_BATCH_SIZE
            warmup_steps = int(steps * WARMUP_PROPORTION)
            # Set evaluation steps.
            if lang_set in ['low', 'medlow']:
                eval_strategy = 'epoch'
                eval_steps = 1
            else:
                eval_strategy = 'steps'
                eval_steps = 1000

            script += """
                export EVAL_STRATEGY={0}
                export EVAL_STEPS={1}
                export MAX_STEPS={2}
                export WARMUP_STEPS={3}
                export N_EXAMPLES_PER_EPOCH={4}
            """.format(eval_strategy, eval_steps, steps, warmup_steps, DATA_QUANTITIES[lang_set])
            # Training.
            langs_str = ' '.join(LANG_SETS[lang_set])
            script += """
                for LANG in {0}; do
            """.format(langs_str)
            # Check if model already exists.
            model_outname = '${{LANG}}_{0}_{1}'.format(monolingual_data_quant_str, model_size)
            model_outpath = os.path.join(args.out_models_dir, model_size, model_outname)
            model_filepath = os.path.join(model_outpath, 'pytorch_model.bin')
            script += """
            if test -f {0}; then
            echo "Model already found; skipping ${{LANG}}."
            continue
            fi
            """.format(model_filepath)
            # Pre-training script.
            if n_devices > 1:
                script += """
                    torchrun --nnodes=1 --nproc_per_node={} --master_port=29401 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \\
                """.format(n_devices)
            else:
                script += """
                    python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \\
                """
            gradient_accumulation_steps = TOTAL_TRAIN_BATCH_SIZE // (batch_per_device * n_devices)
            assert TOTAL_TRAIN_BATCH_SIZE % (batch_per_device * n_devices) == 0
            monolingual_tokenizer_path = os.path.join(args.in_tokenizers_dir, '${LANG}_10k')
            train_path = 'datasets_tokenized_split/${LANG}_train${DATASET_SIZE}.txt'
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
                --train_data_file={5}/${{LANG}}_train{6}.txt \\
                --seed=43 \\
                --override_n_examples=${{N_EXAMPLES_PER_EPOCH}} \\
                --output_dir={7}
                done
            """.format(monolingual_tokenizer_path, model_size,
                    args.data_monolingual_tokenized_dir, batch_per_device,
                    gradient_accumulation_steps, args.data_monolingual_tokenized_dir,
                    monolingual_data_quant_str, model_outpath)

            # Write script to output.
            outfile = codecs.open(script_outpath, 'wb', encoding='utf-8')
            for line in script.split('\n'):
                if line.strip() == '':
                    continue
                outfile.write(line.strip())
                outfile.write('\n')
            outfile.close()
            print('Wrote script: {}'.format(script_outpath))
    print('Done.')


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
