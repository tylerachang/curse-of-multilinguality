# curse-of-multilinguality
Code for the paper, When is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages (2023).
Includes code for preparing pre-training (e.g. tokenizers and data) for multilingual language models.
Language models are pre-trained using: https://github.com/tylerachang/word-acquisition-language-models.
We automate these scripts for our multilingual language modeling experiments.

Pull the code using:
<pre>
git clone https://github.com/tylerachang/curse-of-multilinguality.git
git clone https://github.com/tylerachang/word-acquisition-language-models.git
</pre>
We also use ```terashuf``` for shuffling large datasets:
<pre>
git clone https://github.com/alexandres/terashuf.git
(cd terashuf && make)
</pre>
Python requirements are in ```requirements.txt```. Tested in Python 3.7.

## Dataset sources.
Our text datasets are pulled from a variety of sources:

| Dataset | Link |
| ----- | ----- |
| OSCAR | https://oscar-project.org/ |
| Wikipedia | https://dumps.wikimedia.org/ |
| NLLB | https://huggingface.co/datasets/allenai/nllb |
| NLLB Multi-Domain | https://huggingface.co/datasets/breakend/nllb-multi-domain |
| FLORES-200 | https://github.com/facebookresearch/flores/blob/main/flores200/README.md |
| Leipzig Corpora Collection | https://wortschatz.uni-leipzig.de/en/download/ |
| eBible Translations | https://ebible.org/find/ |
| AfriBERTa | https://huggingface.co/datasets/castorini/afriberta-corpus |
| Nusa | https://huggingface.co/indonlp |
| NusaX | https://huggingface.co/indonlp |
| Languages of Russia | http://web-corpora.net/wsgi3/minorlangs/download |
| Evenki Life Newspaper | https://aclanthology.org/2020.lrec-1.314.pdf |
| Indigenous Languages Corpora | https://github.com/EdTeKLA/IndigenousLanguages_Corpora |
| AmericasNLP 2021 | https://github.com/AmericasNLP/americasnlp2021 |
| AmericasNLP 2022 | https://turing.iimas.unam.mx/americasnlp/2022_st.html |
| AmericasNLP 2023 | https://turing.iimas.unam.mx/americasnlp/2023_st.html |
| AmericasNLI | https://github.com/abteen/americasnli |
| Nunavut Hansard Inuktitut–English Parallel Corpus 3.0 | https://nrc-digital-repository.canada.ca/eng/view/object/?id=c7e34fa7-7629-43c2-bd6d-19b32bf64f60 |
| Fula Speech Corpora | https://huggingface.co/datasets/cawoylel/FulaSpeechCorpora |
| Ewe Language Corpus | https://www.kaggle.com/datasets/yvicherita/ewe-language-corpus |
| Makerere Radio Speech Corpus | https://zenodo.org/records/5855017 |
| English-Luganda Parallel Corpus | https://zenodo.org/records/4764039 |
| Tigrinya Language Modeling Dataset | https://zenodo.org/record/5139094 |
| Lacuna Project: IsiXhosa | https://github.com/Chiamakac/lacuna_pos_ner/tree/main/language_corpus/xho |
| CMU Haitian Creole | http://www.speech.cs.cmu.edu/haitian/text/newswire-all.ht |
| Ulukau | https://ulukau.org/index.php?l=en |
| Cherokee Dictionary | https://www.cherokeedictionary.net/corpus/corpusMain |
| ChrEn | https://huggingface.co/datasets/chr_en |
| Tatoeba | https://huggingface.co/datasets/tatoeba |

All languages and total dataset sizes are listed [here](https://docs.google.com/spreadsheets/d/1rNRLi_2H08T_n3_Iow3M84ml5tAr3cs3xjcFmd91-FA/edit?usp=sharing).

## Tokenize monolingual datasets.
Raw text data should be placed in a folder called ```datasets```.
Each file should be named ```[language_code].txt```.
By default, a SentencePiece tokenizer is trained for each language with maximum vocabulary size 32K and with 10K randomly sampled text lines as training data.
Discussion of tokenization quality is included in our paper.
Datasets are tokenized with token sequence length 128 (concatenating text lines) and maximum 10B sequences per language.
<pre>
python3 curse-of-multilinguality/tokenize_datasets.py
</pre>
Outputs tokenizers to ```tokenizers/monolingual``` and tokenized datasets to ```datasets_tokenized```.

## Create monolingual dataset splits.
First, we shuffle the tokenized datasets.
Outputs to ```datasets_tokenized_shuffled```.
<pre>
mkdir terashuf_tmp_output
export TMPDIR=terashuf_tmp_output
export SEED=42
mkdir datasets_tokenized_shuffled
for FNAME in datasets/*.txt; do
LANG=${FNAME//".txt"/""}
LANG=${LANG//"datasets/"/""}
echo $LANG
terashuf/terashuf < "datasets_tokenized/${LANG}_tokenized.txt" > "datasets_tokenized_shuffled/${LANG}_tokenized.txt"
done
</pre>
Create train and eval splits for monolingual datasets (assumes shuffled datasets).
The last 4K examples (512K tokens) are used as eval sets.
Then, 8*10^k remaining examples are sampled for pre-training (for maximal k).
The smaller pre-training sets (lower k) are iteratively sampled from the pre-training set.
This creates low-resource, ..., high-resource pre-training sets with 8M, 800K, 80K, and 8K examples (1B, 100M, 10M, 1M pre-training tokens).
<pre>
python3 curse-of-multilinguality/split_tokenized_files.py \
--input_dir="datasets_tokenized" \
--output_dir="datasets_tokenized_split"
</pre>

## Prepare pre-training scripts.
These scripts call the pre-training code from https://github.com/tylerachang/word-acquisition-language-models.
The scripts assume the sets of languages in ```utils/constants.py``` (252 languages).

### Monolingual
Prepare scripts to pre-train monolingual baselines.
See preparation script for various default pre-training settings.
For tiny, mini, and small models, the pre-training scripts default to one GPU device for pre-training.
<pre>
python3 curse-of-multilinguality/prepare_monolingual_scripts.py \
--outdir="multilingual_data_prep/monolingual" \
--out_models_dir="models/monolingual" \
--data_monolingual_tokenized_dir="datasets_tokenized_split" \
--in_tokenizers_dir="tokenizers/monolingual"
chmod u+x multilingual_data_prep/monolingual/*.sh
</pre>
To run the monolingual pre-training scripts on one GPU device:
<pre>
export CUDA_VISIBLE_DEVICES=0
multilingual_data_prep/monolingual/train_low_small.sh
</pre>
Replace "low" with language resource amount in \[low, medlow, medhigh, high\], and replace "small" with model size in \[tiny, mini, small\].
This pre-trains the monolingual models for all languages with that dataset and model size.

### Multilingual
Preparing for multilingual pre-training requires additional multilingual tokenization and dataset compilation.
As described in our paper, we vary monolingual dataset size, multilingual dataset size, similarity of the added languages, and model size.
The target language monolingual tokenizer is fixed, and the multilingual data (10 added languages) is tokenized using a shared multilingual tokenizer.
The tokenized datasets are then merged based on shared tokens between the monolingual and multilingual tokenizers.

For computational efficiency, the preparation script runs added language selection (see note in the preparation script on ```lang2vec``` installation if using ```conda```), multilingual tokenizer training, and multilingual tokenization.
For space efficiency (i.e. to save less copies of the added multilingual data per target language), we merge target language monolingual and multilingual datasets on the fly when the pre-training scripts are run.
For details, see notes in the preparation script.

First, we sample 10K lines from each language for tokenizer training.
This avoids having to sample the large datasets every time a multilingual tokenizer is trained.
Samples from ```datasets``` and outputs to ```datasets_10k```.
<pre>
mkdir datasets_10k
python3 curse-of-multilinguality/sample_lines.py
</pre>

Then, prepare scripts to train on the 10 most similar medhigh-resource languages.
This is quite slow, because the multilingual tokenizers are trained and multilingual datasets are tokenized for every target language.
See details in the preparation script header.
<pre>
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
chmod u+x multilingual_data_prep/multilingual_similar/*.sh
</pre>
Similarly, prepare scripts to train on the 10 least similar medhigh-resource languages.
<pre>
python3 curse-of-multilinguality/prepare_multilingual_training.py \
--outdir="multilingual_data_prep/multilingual_dissimilar" \
--out_data_dir="multilingual_data_tokenized/multilingual_dissimilar" \
--out_models_dir="models/multilingual_dissimilar" \
--out_tokenizers_dir="tokenizers/multilingual_dissimilar" \
--added_langs_selection="dissimilar" --added_langs_n=10 \
--added_langs_superset="medhigh" \
--data_text_dir="datasets" --data_tokenizer_text_dir="datasets_10k" \
--data_monolingual_tokenized_dir="datasets_tokenized_split" \
--in_tokenizers_dir="tokenizers/monolingual" \
--lang_similarities_dir="lang_similarities_dir"
chmod u+x multilingual_data_prep/multilingual_dissimilar/*.sh
</pre>
To run the multilingual pre-training scripts:
<pre>
export CUDA_VISIBLE_DEVICES=0
multilingual_data_prep/multilingual_similar/train_low_10similar_medhigh_8k_small.sh
</pre>
Replace "low" with monolingual language resource amount in \[low, medlow, medhigh, high\].
Replace "8k" with amount of added data in each language \[8k, 80k, 800k\] (sequences; corresponding to 10M, 100M, and 1B multilingual tokens total assuming 10 added languages).
Replace "small" with model size in \[tiny, mini, small\].
Replace "similar" with similarity in \[similar, dissimilar\].

## Compiling results.
Compile results (evaluation perplexity for each language model). E.g.:
<pre>
python3 curse-of-multilinguality/compile_results.py \
--input_dir="models/monolingual/small" \
--output_path="results/monolingual_small_results.tsv"
</pre>
To run all compiles, use ```run_compiles.sh```.

## Citation.
<pre>
@article{chang-etal-2023-multilinguality,
  title={When is Multilinguality a Curse? {L}anguage Modeling for 250 High- and Low-Resource Languages},
  author={Tyler A. Chang and Catherine Arnett and Zhuowen Tu and Benjamin K. Bergen},
  journal={arXiv preprint},
  year={2023}
}
</pre>
