__author__ = 'GCassani'

"""Create the co-occurrence vector space from a corpus, given a set of relevant distributional contexts"""


import json
import numpy as np
from time import strftime
from collections import Counter
from context_utils import utterance


def create_test_space(input_corpus, test_perc, contexts, pos_dict, bigrams=True, trigrams=True):

    """
    :param input_corpus:            the same corpus used for training, in the same .json format (see the documentation
                                    to the function collect_contexts for further details
    :param test_perc:               a number indicating the percentage of the input corpus to be used as test set -
                                    ideally this would be 100 - the training percentage, but different values can be
                                    chosen, However, we stress that it is preferable to avoid any overlap between
                                    training and test material
    :param contexts:                a dictionary containing all the contexts collected during training, mapped to the
                                    column index each context has in the training space
    :param pos_dict:                a dictionary mapping CHILDES Parts-of-Speech tags to custom tags (the same that was
                                    used as input to the function collect_contexts
    :param bigrams:                 a boolean indicating whether bigrams are to be collected
    :param trigrams:                a boolean indicating whether trigrams are to be collected
    :return co_occurrences:         a NumPY 2d array, where rows are words, columsn are distributional contexts, and
                                    cells contain integers indicating how many times a word co-occurred with a context
                                    in the input corpus
    :return word_ids                a dictionary mapping words to numerical indices, indicating the corresponding row
                                    in the co-occurrence matrix
    :return word_freqs:             a dictionary mapping words to their frequency count as computed from the test set
    """

    # initialize three dictionaries to keep track of word and context frequency counts, and of the co-occurrences
    # between them
    co_occurrences = np.zeros([0, len(contexts)])
    word_ids = {}
    word_freqs = Counter()
    last_word = 0

    # get the cut-off point where to start considering utterances for the test set
    corpus = json.load(open(input_corpus, 'r+'))
    total_utterances = len(corpus[0])
    cutoff = total_utterances - np.floor(total_utterances / 100 * test_perc)

    # get the indices of utterances marking the 5, 10, 15, ... per cent of the input in order to show progress
    check_points = {np.floor((total_utterances - cutoff) / 100 * n) + cutoff: n for n in np.linspace(5, 100, 20)}

    # set the size of the window in which contexts are collected
    size = 2 if trigrams else 1

    # start considering utterances from the cut-off point computed using the percentage provided as input
    nline = int(cutoff)
    print("Considering utterances for test set from utterance number %d" % cutoff)

    while nline < total_utterances:

        # filter the current utterance removing all words labeled with PoS tags that need to be discarded
        tokens = corpus[0][nline]
        lemmas = corpus[1][nline]
        words = utterance.clean_utterance(tokens, lemmas=lemmas, pos_dict=pos_dict)

        # if at least one valid word was present in the utterance and survived the filtering step, collect all possible
        # contexts from the utterance, as specified by the input granularities
        if len(words) > 1:
            words.append('#end~bound')
            last_idx = len(words) - 1
            idx = 1
            # first and last element are dummy words for sentence boundaries
            while idx < last_idx:
                # using all words as pivots, collect all possible contexts, check which ones were also collected
                # during training and keep track of word-context co-occurrences involving only this subset of contexts,
                # updating the counts
                context_window = utterance.construct_window(words, idx, size)
                current_contexts = utterance.get_ngrams(context_window, bigrams=bigrams, trigrams=trigrams)

                target_word = words[idx]
                word_freqs[target_word] += 1
                # the frequency count is incremented once. However, the word is counted as occurring with all the
                # contexts of the window, so it may be the case that a word has a higher diversity count than frequency
                # count. This is not an error, but the result of harvesting more than one context for every occurrence
                # of a word.

                if target_word not in word_ids:
                    word_ids[target_word] = last_word
                    last_word += 1
                    new_row = np.zeros([1, co_occurrences.shape[1]])
                    co_occurrences = np.vstack([co_occurrences, new_row])

                for context in current_contexts:
                    if context in contexts:
                        row_idx = word_ids[target_word]
                        col_idx = contexts[context]
                        co_occurrences[row_idx, col_idx] += 1
                idx += 1

        # print progress
        if nline in check_points:
            print(strftime("%Y-%m-%d %H:%M:%S") +
                  ": %d%% of the utterances allocated as test set has been processed." % check_points[nline])

        nline += 1

    return co_occurrences, word_ids, word_freqs
