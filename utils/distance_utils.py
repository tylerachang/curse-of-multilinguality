"""
Utilities for computing distances and similarities, e.g. between languages.
"""

import os
from tqdm import tqdm
import numpy as np
import itertools
import scipy.spatial
import codecs

from utils.data_utils import get_vocab
from utils.constants import LANG_SETS


# Caches to lang_similarities_dir.
# Returns similarity matrices for different metrics.
# Finds language tokenizers by replacing [lang] in tokenizer_template.
# Only uses tokenizer_template if the similarities have not been cached.
# Only requires lang2vec if the similarities have not been cached.
# If a language is not available in lang2vec, the similarities except for
# vocab similarity are set to np.nan.
def get_lang_similarities(langs, lang_similarities_dir, tokenizer_template):
    lang_key_outpath = os.path.join(lang_similarities_dir, 'lang_key.txt')
    syntax_outpath = os.path.join(lang_similarities_dir, 'syntax.npy')
    phonology_outpath = os.path.join(lang_similarities_dir, 'phonology.npy')
    inventory_outpath = os.path.join(lang_similarities_dir, 'inventory.npy')
    geo_outpath = os.path.join(lang_similarities_dir, 'geo.npy')
    raw_vocab_outpath = os.path.join(lang_similarities_dir, 'raw_vocab.npy')
    vocab_outpath = os.path.join(lang_similarities_dir, 'vocab.npy')
    if os.path.isfile(lang_key_outpath):
        # Read langs.
        infile = codecs.open(lang_key_outpath, 'rb', encoding='utf-8')
        saved_langs = infile.read()
        saved_langs = saved_langs.strip().split()
        infile.close()
        assert saved_langs == langs, 'Saved similarities use a different set of languages.'
        syntax = np.load(syntax_outpath)
        phonology = np.load(phonology_outpath)
        inventory = np.load(inventory_outpath)
        geo = np.load(geo_outpath)
        vocab = np.load(vocab_outpath)
        return syntax, phonology, inventory, geo, vocab
    # Compute similarities.
    print('Getting language vectors with lang2vec.')
    # Only attempt this import if necessary, because not available in conda.
    import lang2vec.lang2vec as l2v
    syntax_vecs = dict()
    phonology_vecs = dict()
    inventory_vecs = dict()
    geo_vecs = dict()
    for lang in tqdm(langs):
        iso3 = lang.split('_')[0]
        try:
            syntax_vecs[lang] = np.array(l2v.get_features(iso3, 'syntax_knn')[iso3])
            phonology_vecs[lang] = np.array(l2v.get_features(iso3, 'phonology_knn')[iso3])
            inventory_vecs[lang] = np.array(l2v.get_features(iso3, 'inventory_knn')[iso3])
            geo_vecs[lang] = np.array(l2v.get_features(iso3, 'geo')[iso3])
        except Exception as e:
            # Language not supported by lang2vec.
            error_message = str(e)
            assert 'lang2vec.available_languages()' in error_message
            print('No lang2vec for language: {}'.format(lang))
            syntax_vecs[lang] = np.array([np.nan])
            phonology_vecs[lang] = np.array([np.nan])
            inventory_vecs[lang] = np.array([np.nan])
            geo_vecs[lang] = np.array([np.nan])
    print('Getting language vocabularies.')
    vocabs = dict()
    for lang in tqdm(langs):
        tokenizer_path = tokenizer_template.replace('[lang]', lang)
        vocab = get_vocab(tokenizer_path, as_set=True)
        vocabs[lang] = vocab
    print('Computing language similarities.')
    syntax = np.nan * np.ones((len(langs), len(langs)))
    phonology = np.nan * np.ones((len(langs), len(langs)))
    inventory = np.nan * np.ones((len(langs), len(langs)))
    geo = np.nan * np.ones((len(langs), len(langs)))
    vocab = np.nan * np.ones((len(langs), len(langs)))
    pairs = itertools.combinations(range(len(langs)), 2)
    for lang_i, lang_j in tqdm(pairs):
        lang1 = langs[lang_i]
        lang2 = langs[lang_j]
        # Lang2vec similarities. These are nan if one vector is nan (i.e. not found
        # in lang2vec).
        sim = 1.0 - scipy.spatial.distance.cosine(syntax_vecs[lang1], syntax_vecs[lang2])
        syntax[lang_i, lang_j] = sim
        syntax[lang_j, lang_i] = sim
        sim = 1.0 - scipy.spatial.distance.cosine(phonology_vecs[lang1], phonology_vecs[lang2])
        phonology[lang_i, lang_j] = sim
        phonology[lang_j, lang_i] = sim
        sim = 1.0 - scipy.spatial.distance.cosine(inventory_vecs[lang1], inventory_vecs[lang2])
        inventory[lang_i, lang_j] = sim
        inventory[lang_j, lang_i] = sim
        sim = 1.0 - scipy.spatial.distance.cosine(geo_vecs[lang1], geo_vecs[lang2])
        geo[lang_i, lang_j] = sim
        geo[lang_j, lang_i] = sim
        # Vocab overlap. Normalize by 1 because we will take the log.
        vocab1 = vocabs[lang1]
        vocab2 = vocabs[lang2]
        overlap = len(vocab1.intersection(vocab2)) + 1
        vocab[lang_i, lang_j] = overlap
        vocab[lang_j, lang_i] = overlap
    # Save raw vocab.
    os.makedirs(lang_similarities_dir, exist_ok=True)
    np.save(raw_vocab_outpath, vocab, allow_pickle=False)
    # Log.
    vocab = np.log(vocab)
    # Normalize.
    syntax = (syntax - np.nanmean(syntax)) / np.nanstd(syntax)
    phonology = (phonology - np.nanmean(phonology)) / np.nanstd(phonology)
    inventory = (inventory - np.nanmean(inventory)) / np.nanstd(inventory)
    geo = (geo - np.nanmean(geo)) / np.nanstd(geo)
    vocab = (vocab - np.nanmean(vocab)) / np.nanstd(vocab)
    # Save arrays and languages.
    print('Saving in: {}'.format(lang_similarities_dir))
    np.save(syntax_outpath, syntax, allow_pickle=False)
    np.save(phonology_outpath, phonology, allow_pickle=False)
    np.save(inventory_outpath, inventory, allow_pickle=False)
    np.save(geo_outpath, geo, allow_pickle=False)
    np.save(vocab_outpath, vocab, allow_pickle=False)
    outfile = codecs.open(lang_key_outpath, 'wb', encoding='utf-8')
    for lang in langs:
        outfile.write(lang)
        outfile.write('\n')
    outfile.close()
    print('Saved.')
    return syntax, phonology, inventory, geo, vocab


# Returns n_langs similar languages from lang_set.
# Uses the cached language similarities in lang_similarities_dir if possible,
# in which case tokenizer_template is ignored. Reverse=True will return
# dissimilar languages.
def get_similar_languages(target_lang, lang_set, n_langs, lang_similarities_dir,
                          reverse=False, tokenizer_template='[lang]_10k',
                          all_langs=LANG_SETS['low']):
    # Retrieve similarities for all languages (i.e. the set of languages with
    # at least low-resource data).
    sims = get_lang_similarities(all_langs, lang_similarities_dir, tokenizer_template)
    syntax, _, _, geo, vocab = sims
    # Impute means.
    syntax[np.isnan(syntax)] = np.nanmean(syntax)
    geo[np.isnan(geo)] = np.nanmean(geo)
    vocab[np.isnan(vocab)] = np.nanmean(vocab)
    # Use mean similarity over syntax, vocabulary, and geography.
    average_sims = np.mean(np.stack([syntax, vocab, geo], axis=0), axis=0)
    # Similarities to the target language.
    lang_i = all_langs.index(target_lang)
    lang_sims = dict() # Map from languages in lang_set to their similarity with lang.
    for lang2 in lang_set:
        lang_j = all_langs.index(lang2)
        sim = average_sims[lang_i, lang_j]
        lang_sims[lang2] = sim
    sorted_langs = sorted(lang_sims.keys(), key=lang_sims.get)
    # Remove target language:
    sorted_langs = [lang2 for lang2 in sorted_langs if lang2 != target_lang]
    # By default, languages are in order of increasing similarity.
    if reverse:
        # Dissimilar languages.
        return sorted_langs[:n_langs]
    else:
        # Similar languages.
        return list(reversed(sorted_langs))[:n_langs]
