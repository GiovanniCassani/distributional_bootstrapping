import json
import numpy as np
from time import strftime
from context_utils.vector_spaces.maker import sort_items
from context_utils.corpus import get_words_and_contexts
from context_utils import utterance


def collect_contexts(input_corpus, training_perc, pos_dict, bigrams=True, trigrams=True):

    """
    :param input_corpus:    a .json file containing transcripts of child-caregiver interactions extracted from the
                            CHILDES database. The json file consists of two lists of lists, of the same length, both
                            contain utterances but encoded differently. The first encodes each utterance as a list
                            of tokens; the second encodes each utterance as a list of lemmas and Part-of-Speech
                            tags, joined by a tilde ('~').
    :param training_perc:   a number indicating the percentage of the input corpus to be used for training
    :param pos_dict:        a dictionary mapping CHILDES Parts-of-Speech tags to custom tags
    :param bigrams:         a boolean indicating whether bigrams are to be collected
    :param trigrams:        a boolean indicating whether trigrams are to be collected
    :return co_occurrences: a NumPy 2d array, where each row is a word and each column is a context. Each cell contains
                            an integer specifying how many times a word and a context co-occurred in the input corpus
    :return context_ids:    a dictionary mapping strings denoting contexts to their column index in the co-occurrence
                            matrix
    :return word_ids:       a dictionary mapping strings denoting words to their row index in the co-occurrence matrix

    This function collects all distributional contexts satisfying the input criterion scanning the input corpus.
    """

    words = set()
    contexts = set()

    # get the cut-off point where to stop training
    corpus = json.load(open(input_corpus, 'r+'))
    total_utterances = len(corpus[0])
    cutoff = np.floor(total_utterances / float(100) * training_perc)
    print("Considering utterances for training set until utterance number %d" % cutoff)

    # set the size of the window in which contexts are collected
    size = 2 if trigrams else 1
    # initialize a list where cleaned utterances will be stored for re-use
    filtered_corpus = []
    # initialize a counter to keep track of how many lines have been processed
    n_line = 0
    # since utterance boundaries are always added to utterances, a non-empty utterance has more than 2 elements
    min_length = 2

    # process lines until the cut-off is reached
    while n_line < cutoff:

        # filter the current utterance removing all words labeled with PoS tags that need to be discarded
        # the clean_utterance function automatically adds beginning and end of utterance markers to the utterance,
        # which is return as a list of strings
        tokens = corpus[0][n_line]
        lemmas = corpus[1][n_line]
        w, c = get_words_and_contexts(tokens, filtered_corpus, min_length, size,
                                      lemmas=lemmas, pos_dict=pos_dict, bigrams=bigrams, trigrams=trigrams)
        words = words.union(w)
        contexts = contexts.union(c)
        n_line += 1

    words2ids = sort_items(words)
    contexts2ids = sort_items(contexts)

    print(strftime("%Y-%m-%d %H:%M:%S") + ": I collected all words and contexts in the test corpus.")
    print()

    co_occurrences = np.zeros([len(words2ids), len(contexts2ids)])

    total_utterances = len(filtered_corpus)
    check_points = {np.floor(total_utterances / float(100) * n): n for n in np.linspace(5, 100, 20)}

    word_frequencies = np.zeros(len(words2ids))

    n_line = 0
    for u in filtered_corpus:
        idx = 1
        last_idx = len(u) - 1
        while idx < last_idx:
            # using all valid words as pivots, collect all possible contexts and keep track of word-context
            # co-occurrences, updating the counts
            current_word = u[idx]
            context_window = utterance.construct_window(u, idx, size)
            current_contexts = utterance.get_ngrams(context_window, bigrams=bigrams, trigrams=trigrams)
            row_idx = words2ids[current_word]
            word_frequencies[row_idx] += 1
            for context in current_contexts:
                col_idx = contexts2ids[context]
                co_occurrences[row_idx, col_idx] += 1
            idx += 1
        n_line += 1

        # print progress
        if n_line in check_points:
            print(strftime("%Y-%m-%d %H:%M:%S") +
                  ": %d%% of the utterances allocated as training corpus has been processed." % check_points[n_line])

    return co_occurrences, contexts2ids, words2ids, word_frequencies