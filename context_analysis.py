__author__ = 'GCassani'

import os
import json
import scipy.stats
import operator
import argparse
import numpy as np
import context_utils as ctxt
from collections import defaultdict, Counter
from time import strftime


def table2dict(in_file, sep, header=True, c=3, t=50):

    """
    :param in_file: the path to a .txt file containing several columns, where the first one consists of unique items
                    that can be therefore used as keys in a dictionary
    :param sep:     a string indicating the character separating columns in the input file
    :param header:  a boolean specifying whether the input file has column headers that specify column names
    :param c:       the column number (1-indexed) containing the information about the percentage of the corpus analyzed
    :param t:       a scalar indicating at which percentage of the corpus the function stops considering elements
    :return d:      a dictionary where the items in the first column of the input file are used as keys and the items in
                    all other columns are concatenated using the separator character and assigned to the corresponding
                    key as value
    """

    d = defaultdict(dict)
    col_names = {}

    with open(in_file, 'r') as f:

        line = f.readline()
        record = line.strip().split(sep)
        for idx, field in enumerate(record):
            if header:
                col_names[idx] = field
            else:
                col_names[idx] = "col" + str(idx)
                d[record[idx]][col_names[idx]] = record[idx]

        for line in f:
            record = line.strip().split(sep)
            try:
                if int(record[c - 1]) == t:
                    for i in range(len(record)):
                        d[record[0]][col_names[i]] = record[i]
            except IndexError:
                pass

    return d, col_names


########################################################################################################################


def intersect(corpora, wd, file_ext, sep='\t'):

    """
    :param corpora:     an iterable containing strings indicating the corpora's names to be considered
    :param wd:          a string pointing to the directory where files to be accessed are stored
    :param file_ext:    a string indicating how all files to be accessed end, after the corpus name
    :param sep:         the character separating the different fields in the input files
    :return shared:     a set containing the words that are common to all input files

    This function takes summary tables created running the main function in this module, for both contexts and words:
    it reads in the first input file, store everything in a dictionary using words or contexts as keys (depending on the
    value of file_ext). Iteratively, all input files are read in, the words or contexts in them are stored and only the
    items that also appeared in all other corpora analyzed so far are retained. In the end, the function gives the set
    of items (words or contexts) that appear in the summary table of all corpora provided as input.

    The input files should all be stored in the same directory, begin with the corpus name, and end with the same
    string, e.g. '/User/Folder1/Luke_wordStats.txt': '/User/Folder1' is wd, 'Luke' is the corpus name, and
    '_wordStats.txt' is the file_ext, i.e. whatever comes after the corpus name.
    """

    shared = set()
    for corpus in corpora:
        infile = os.path.join(wd, ''.join([corpus, file_ext]))
        d, names = table2dict(infile, sep)
        if not shared:
            shared = set(d.keys())
        else:
            shared = set(d.keys()).intersection(shared)

    return shared


########################################################################################################################


def print_targets(targets, output, corpora, wd, file_ext, sep='\t', header=True):

    """
    :param targets:     an iterable containing strings indicating the items in the input files to be retained
    :param output:      the name of the file where the function will print its output
    :param corpora:     an iterable containing strings indicating the corpora's names to be considered
    :param wd:          a string indicating the directory containing all input files
    :param file_ext:    a string indicating how all files to be accessed end, after the corpus name
    :param sep:         the character separating the different fields in the input files
    :param header:      a boolean specifying whether the input files have a header, which will be printed at the top
                        of the output file
    """

    if not wd.endswith("/"):
        wd += "/"

    flag = 0

    with open(wd + output, "a+") as fo:
        for corpus in corpora:
            infile = wd + corpus + file_ext
            with open(infile, "r") as fi:
                if header and not flag:
                    colnames = fi.readline()
                    fo.write(colnames)
                    flag = 1

                for line in fi:
                    target = line.strip().split(sep)[0]
                    if target in targets:
                        fo.write(line)


########################################################################################################################


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
        w, c = ctxt.words_and_contexts(tokens, filtered_corpus, min_length, size,
                                       lemmas=lemmas, pos_dict=pos_dict, bigrams=bigrams, trigrams=trigrams)
        words = words.union(w)
        contexts = contexts.union(c)
        n_line += 1

    words2ids = ctxt.sort_items(words)
    contexts2ids = ctxt.sort_items(contexts)

    print(strftime("%Y-%m-%d %H:%M:%S") + ": I collected all words and contexts in the test corpus.")
    print()

    co_occurrences = np.zeros([len(words2ids), len(contexts2ids)])

    total_utterances = len(filtered_corpus)
    check_points = {np.floor(total_utterances / float(100) * n): n for n in np.linspace(5, 100, 20)}

    word_frequencies = np.zeros(len(words2ids))

    n_line = 0
    for utterance in filtered_corpus:
        idx = 1
        last_idx = len(utterance) - 1
        while idx < last_idx:
            # using all valid words as pivots, collect all possible contexts and keep track of word-context
            # co-occurrences, updating the counts
            current_word = utterance[idx]
            context_window = ctxt.construct_window(utterance, idx, size)
            current_contexts = ctxt.get_ngrams(context_window, bigrams=bigrams, trigrams=trigrams)
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


########################################################################################################################


def compute_context_statistics(co_occurrences, word_frequencies, context_ids, feature_weights, output_file, perc):

    """
    :param co_occurrences:      a NumPy 2d array, where each row is a word and each column is a context. Each cell
                                contains an integer specifying how many times a word and a context co-occurred in the
                                input corpus
    :param word_frequencies:    a 1d NumPy array containing word frequency counts (aligned with the row ids from the
                                co-occurrence matrix: the first number in word_frequencies is the frequency of the word
                                in the first row of the co-occurrence matrix
    :param context_ids:         a dictionary mapping contexts to column indices (to access columns in the co-occurrence
                                matrix)
    :param feature_weights:     a dictionary mapping contexts to their weights computed using TiMBL: weights tell how
                                useful a context is for categorizing words. Better contexts have larger weights
    :param output_file:         a string indicating a .json file where a dictionary mapping a context to several
                                statistics about it is stored
    :param perc:                a number indicating the percentage of utterances allocated to the training file
    :return context_statistics: a dictionary where each context is used as key an is mapped to several properties that
                                are computed within the function

    This function computes distributional statistics concerning the contexts collected by the function collect_contexts:
    # - granularity (how many words does a context consist of)
    # - position of the empty slot (left, right, non-adjacent)
    # - token frequency
    # - type frequency
    # - average conditional probability of the context given all the words it co-occurs with
    # - normalized entropy of the context over the words it co-occurs with
    """

    corpus = os.path.basename(output_file).split("_")[0]

    context_statistics = defaultdict(dict)
    # for each context, compute the statistics mentioned in the helper
    for context, idx in sorted(context_ids.items(), key=operator.itemgetter(1)):

        # each column vector contains the co-occurrence counts of a context and all the words
        word_counts = co_occurrences[:, idx]

        gr = feature_weights[idx]['gr']
        ig = feature_weights[idx]['ig']
        x2 = feature_weights[idx]['x2']
        sv = feature_weights[idx]['sv']

        # token frequency of a context is the logged sum of all the co-occurrence counts in the column
        token_frequency = np.log10(sum(word_counts))

        # type frequency of a context is the logged number of non-zero elements in the column, corresponding to the
        # number of words the context co-occurs with
        type_frequency = np.log10(np.count_nonzero(word_counts))

        # average conditional probability of context given words is computed by first dividing the co-occurrence count
        # between a context and a word by the frequency count of the word alone. This step is carried out for all words
        # a context occurred with, i.e. the non-zero cells of the column being considered, summing the probabilities and
        # then dividing by the number of words the context occurred with
        conditional_probabilities = np.zeros(np.count_nonzero(word_counts))
        for i, row in enumerate(np.nonzero(word_counts)[0]):
            word_frequency = word_frequencies[row]
            conditional_probabilities[i] = co_occurrences[row, idx] / word_frequency
        conditional_probability = np.average(conditional_probabilities)

        # normalized entropy is computed by first extracting all non-zero co-occurrence values from a column vector from
        # the co-occurrence matrix, and then taking the entropy of this vector using the number of non-zero elements in
        # the vector as the log base.
        nonzero_counts = word_counts[np.nonzero(word_counts)]
        if len(nonzero_counts) > 1:
            norm_entropy = scipy.stats.entropy(nonzero_counts, base=len(nonzero_counts))
        elif len(nonzero_counts) == 1:
            norm_entropy = 0
        else:
            norm_entropy = 1

        # the number of constituent words is inferred using the number of underscores used to create the context in the
        # first place
        constituents = len(context.split("__"))
        idx_slot = context.split("__").index("X")
        if idx_slot == 0:
            kind = 'right'
        elif idx_slot + 1 == constituents:
            kind = 'left'
        else:
            kind = 'non-adjacent'

        context_statistics[context] = {'corpus': corpus,
                                       'perc': perc,
                                       'gr': gr,
                                       'ig': ig,
                                       'x2': x2,
                                       'sv': sv,
                                       'token frequency': token_frequency,
                                       'type frequency': type_frequency,
                                       'conditional probability': conditional_probability,
                                       'normalized entropy': norm_entropy,
                                       'constituents': constituents,
                                       'type': kind}

    # write everything to a .txt file
    with open(output_file, "a+") as f:
        if os.stat(output_file).st_size == 0:
            f.write("\t".join(["Context", "Corpus", "Perc", "Gain_Ratio", "Info_Gain", "Xsquared",
                               "Shared_var", "Freq", "Div", "Cond_Prob", "Norm_Entr", "Const", "Type"]))
        f.write("\n")
        for context in context_statistics:
            f.write("\t".join([context,
                               context_statistics[context]['corpus'],
                               str(context_statistics[context]['perc']),
                               str(context_statistics[context]['gr']),
                               str(context_statistics[context]['ig']),
                               str(context_statistics[context]['x2']),
                               str(context_statistics[context]['sv']),
                               str(context_statistics[context]['token frequency']),
                               str(context_statistics[context]['type frequency']),
                               str(context_statistics[context]['conditional probability']),
                               str(context_statistics[context]['normalized entropy']),
                               str(context_statistics[context]['constituents']),
                               str(context_statistics[context]['type'])]))
            f.write("\n")

    return context_statistics


########################################################################################################################


def compute_feature_weights(training_space, output_file):

    """
    :param training_space:  the path to a file containing the co-occurrence count matrix derived from the training
                            corpus
    :param output_file:     the path to a file where the weight of each context will be written
    :return weights:        a dictionary mapping each context to 4 strings, each indicating one of the possible
                            weighting schemes: gain ratio ('gr'), information gain ('ig'), X-square ('x2'), and shared
                            variance ('sv'). Each string map to the weight of the corresponding contexts under the
                            weighting scheme at hand. All scores are stored for later processing.
    """

    weights = defaultdict(dict)

    with open(training_space, 'r') as t:
        first_line = t.readline()
        n = len(first_line.strip().split("\t")) + 100

    train_space = ' -f ' + training_space
    out_file = ' -W ' + output_file
    timbl_cmd = 'timbl -mN:I1 -N' + str(n) + train_space + out_file
    print(timbl_cmd)
    os.system(timbl_cmd)

    with open(output_file, "r") as f:

        gr, ig, x2, sv = [0, 0, 0, 0]

        for line in f:

            if line.strip() == '# gr':
                gr, ig, x2, sv = [1, 0, 0, 0]
            elif line.strip() == '# ig':
                gr, ig, x2, sv = [0, 1, 0, 0]
            elif line.strip() == '# x2':
                gr, ig, x2, sv = [0, 0, 1, 0]
            elif line.strip() == '# sv':
                gr, ig, x2, sv = [0, 0, 0, 1]

            if any([gr, ig, x2, sv]):
                try:
                    feature, weight = line.strip().split("\t")
                    if gr:
                        weights[int(feature) - 2]['gr'] = float(weight)
                    elif ig:
                        weights[int(feature) - 2]['ig'] = float(weight)
                    elif x2:
                        weights[int(feature) - 2]['x2'] = float(weight)
                    elif sv:
                        weights[int(feature) - 2]['sv'] = float(weight)
                except ValueError:
                    pass

    return weights


########################################################################################################################


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
        words = ctxt.clean_utterance(tokens, lemmas=lemmas, pos_dict=pos_dict)

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
                context_window = ctxt.construct_window(words, idx, size)
                current_contexts = ctxt.get_ngrams(context_window, bigrams=bigrams, trigrams=trigrams)

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


########################################################################################################################


def compute_words_statistics(co_occurrences, word_ids, word_frequencies, accuracies,
                             feature_weights, output_file, perc):

    """
    :param co_occurrences:      a NumPy 2d array where rows are words, columns are distributional contexts and cells
                                contain integers indicating how many times a word co-occurred with each context in the
                                input corpus
    :param word_ids:            a dictionary mapping words to their row indices in the co-occurrence matrix
    :param word_frequencies:    a 1d NumPy array containing word frequency counts (aligned with the row ids from the
                                co-occurrence matrix: the first number in word_frequencies is the frequency of the word
                                in the first row of the co-occurrence matrix
    :param accuracies:          a dictionary mapping each word to the correct PoS tag and two experiments ('sklearn' and
                                'timbl') and the corresponding:
                                - classification outcome ('accuracy'): 1 correct, 0 otherwise
                                - the predicted PoS tag by the experiment ('predicted')
                                As an example, for the word 'dog' you have something like this
                                - accuracies['dog']:    'correct':  N
                                                        'sklearn':  'predicted':    N
                                                                    'accuracy':     1
                                                        'timbl':    'predicted':    ADJ
                                                                    'accuracy:      1
                                where the correct tag is Noun, sklearn correctly predicted N (hence accuracy is 1) but
                                timbl predicted ADJective (hence accuracy is 0)
    :param feature_weights:     a dictionary mapping context indices to their weights computed using TiMBL
    :param output_file:         a string pointing to a .json file where the dictionary containing the statistics
                                computed for each word will be stored
    :param perc:                a number indicating the percentage of utterances allocated to the training file
    :return word_statistics:    a dictionary containing the statistics computed for each word

    This function computes several distributional statistics for words:
    - token frequency
    - number of contexts it co-occurs with
    - average conditional probability of a word given all the contexts it co-occurs with
    - normalized entropy of the word on the contexts it occurs with
    """

    corpus = os.path.basename(output_file).split("_")[0]

    word_statistics = defaultdict(dict)

    for word in accuracies:

        # each row vector contains the co-occurrence counts for a word over all the contexts
        idx = word_ids[word]
        context_counts = co_occurrences[idx, :]

        # accuracy is a binary variable, 1 for correctly categorized words, 0 otherwise
        try:
            sklearn_accuracy = accuracies[word]['sklearn']['accuracy']
            sklearn_predicted = accuracies[word]['sklearn']['predicted']
        except KeyError:
            sklearn_accuracy = float('nan')
            sklearn_predicted = '-'

        try:
            timbl_accuracy = accuracies[word]['timbl']['accuracy']
            timbl_predicted = accuracies[word]['timbl']['predicted']
        except KeyError:
            timbl_accuracy = float('nan')
            timbl_predicted = '-'

        token_frequency = np.log10(word_frequencies[idx])

        # the number of contexts a word occurs in is the cardinality of the set of contexts with which the word
        # co-occurred in the test set
        context_diversity = np.count_nonzero(context_counts)
        context_diversity = np.log10(context_diversity) if context_diversity else 0

        gain_ratios = []
        info_gains = []
        shared_vars = []
        xsquares = []
        if context_diversity:
            context_indices = np.nonzero(context_counts)
            for index in context_indices[0]:
                gain_ratios.append(feature_weights[index]['gr'])
                info_gains.append(feature_weights[index]['ig'])
                xsquares.append(feature_weights[index]['x2'])
                shared_vars.append(feature_weights[index]['sv'])

        gain_ratios = np.asarray(gain_ratios)
        avg_gr = np.mean(gain_ratios)
        med_gr = np.median(gain_ratios)

        info_gains = np.asarray(info_gains)
        avg_ig = np.mean(info_gains)
        med_ig = np.median(info_gains)

        shared_vars = np.asarray(shared_vars)
        avg_sv = np.mean(shared_vars)
        med_sv = np.median(shared_vars)

        xsquares = np.asarray(xsquares)
        avg_x2 = np.mean(xsquares)
        med_x2 = np.median(xsquares)

        # the average conditional probability of word given the contexts it co-occurred with is computed by dividing the
        # co-occurrence count of a word and a context by the context frequency. This step is carried out for all
        # contexts the word occurred with, summing the probabilities and finally averaging.
        conditional_probabilities = np.zeros(0)
        for col in range(co_occurrences.shape[1]):
            if co_occurrences[idx, col] != 0:
                context_frequency = sum(co_occurrences[:, col])
                curr_probability = co_occurrences[idx, col] / context_frequency
                conditional_probabilities = np.append(conditional_probabilities, curr_probability)
        if len(conditional_probabilities):
            conditional_probability = sum(conditional_probabilities) / len(conditional_probabilities)
        else:
            conditional_probability = 0

        # normalized entropy captures the skeweness of the distribution of the co-occurrence counts of a word and the
        # contexts it co-occurred with. The logbase is set to the number of contexts the word occurred with to have a
        # score within 0 and 1 regardless of the number of contexts, where 1 is maximum entropy
        nonzero_counts = context_counts[np.nonzero(context_counts)]
        if len(nonzero_counts) > 1:
            norm_entropy = scipy.stats.entropy(nonzero_counts, base=len(nonzero_counts))
        elif len(nonzero_counts) == 1:
            norm_entropy = 0
        else:
            norm_entropy = 1

        word_statistics[word] = {'corpus': corpus,
                                 'perc': perc,
                                 'correct': accuracies[word]['correct'],
                                 'timbl_predicted': timbl_predicted,
                                 'timbl_accuracy': timbl_accuracy,
                                 'sklearn_predicted': sklearn_predicted,
                                 'sklearn_accuracy': sklearn_accuracy,
                                 'token frequency': token_frequency,
                                 'context diversity': context_diversity,
                                 'conditional probability': conditional_probability,
                                 'normalized entropy': norm_entropy,
                                 'avg_gr': avg_gr,
                                 'med_gr': med_gr,
                                 'avg_ig': avg_ig,
                                 'med_ig': med_ig,
                                 'avg_sv': avg_sv,
                                 'med_sv': med_sv,
                                 'avg_x2': avg_x2,
                                 'med_x2': med_x2}

    print()

    # print all information to a .txt file
    with open(output_file, "a+") as f:
        if os.stat(output_file).st_size == 0:
            f.write("\t".join(["Word", "Corpus", "Perc", "Correct", "timbl_predicted", "timbl_accuracy",
                               "sklearn_predicted", "sklearn_accuracy", "Freq", "Div", "Cond_Prob",
                               "Norm_Entr", "avg_gr", "med_gr", "avg_ig", "med_ig", "avg_sv", "med_sv",
                               "avg_x2", "med_x2"]))
            f.write("\n")
        for word in word_statistics:
            f.write("\t".join([word,
                               word_statistics[word]['corpus'],
                               str(word_statistics[word]['perc']),
                               word_statistics[word]['correct'],
                               word_statistics[word]['timbl_predicted'],
                               str(word_statistics[word]['timbl_accuracy']),
                               word_statistics[word]['sklearn_predicted'],
                               str(word_statistics[word]['sklearn_accuracy']),
                               str(word_statistics[word]['token frequency']),
                               str(word_statistics[word]['context diversity']),
                               str(word_statistics[word]['conditional probability']),
                               str(word_statistics[word]['normalized entropy']),
                               str(word_statistics[word]['avg_gr']),
                               str(word_statistics[word]['med_gr']),
                               str(word_statistics[word]['avg_ig']),
                               str(word_statistics[word]['med_ig']),
                               str(word_statistics[word]['avg_sv']),
                               str(word_statistics[word]['med_sv']),
                               str(word_statistics[word]['avg_x2']),
                               str(word_statistics[word]['med_x2'])]))
            f.write("\n")

    return word_statistics


########################################################################################################################


def context_analysis(input_corpus, pos_dict, output_dir, leave_one_out=False, training_perc=70, test_perc=30,
                     bigrams=True, trigrams=True, nn=1, diag_value=None, sklearn=True, timbl=False):

    """
    :param input_corpus:    a .json file consisting of a list of lists: each utterance is a list of two lists, the first
                            consists of tokens, the second of corresponding lemmas and PoS tags
    :param pos_dict:        a dictionary mapping CHILDES PoS tags to custom tags
    :param output_dir:      the path of the directory where all the results of the simulation will be printed
    :param leave_one_out:   a boolean to specify whether to use a leave-one-out setting where words from the training
                            set are categorized searching the same set of words (except for the target) for the nearest
                            neighbours. Default is false, meaning that a train-test approach is used
    :param training_perc:   the percentage of the input corpus to be used for training
    :param test_perc:       the percentage of the input corpus to be used for testing (if theleave_one_out parameter is
                            set to False)
    :param bigrams:         a boolean specifying whether to consider bigrams or not
    :param trigrams:        a boolean specifying whether to consider trigrams or not
    :param nn:              the number of nearest neighbours to consider when categorizing a new item
    :param diag_value:      the value of the cells on the main diagonal in the matrix of similarities constructed to
                            retrieve nearest neighbours: default to None means that the diagonal is left untouched. 
                            However, setting it to 0 excludes those cells from the possible nearest neighbours: cells on 
                            the diagonal contain the similarity between the vectors of the same word type, one collected
                            in the training corpus, the other in the test corpus. Even though the two vectors are not 
                            identical, it can still be expected that they will be more similar than vectors for 
                            different types. Thus, one can artificially exclude same types from the pool of nearest 
                            neighbours. This is done automatically in the leave-one-out setting, since the vectors would
                            be identical, being collected in the same corpus.
    :param sklearn:         a boolean specifying whether to perform the categorization experiment using the sklearn
                            implementation that relies on cosine as a distance metric
    :param timbl:           a boolean specifying whether to perform the categorization experiment using the TiMBL 
                            implementation that relies on numeric overlap as a distance metric
    """

    if sum([sklearn, timbl]) == 0:
        raise ValueError("Please select a classification experiment by setting the sklearn parameter, the timbl"
                         "parameter, or both to True.")

    # get the name of the corpus and create corpus specific file-names to store the derived co-occurrence matrix,
    # the statistics derived for the contexts, the statistics derived for the words, the weights computed for the files
    # specify the file extension of all these files, except the plot of the cosine similarities, which is gonna be saved
    # in .pdf
    ext = '.csv'
    corpus_name = os.path.splitext(os.path.basename(input_corpus))[0]
    similarities_plot = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'similarities']),
                                                          str(training_perc), '.pdf']))
    training_space = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'training']), str(training_perc), ext]))
    context_output = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'contextStats']), ext]))
    weights_file = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'featWeights']), str(training_perc), ext]))

    print()
    print(strftime("%Y-%m-%d %H:%M:%S") + ": Started creating the vector space from the training corpus.")

    # traverse the input corpus until the specified percentage that should be allocated for training:
    # derive a word-context co-occurrence matrix where rows are words and columns are contexts (all the unique rows and
    # unique contexts from the input corpus, mapped to numerical indices).
    # - training_co_occurrences is a 2dNumPy array storing word-context co-occurrences
    # - context_ids is a dictionary mapping contexts to numerical indices pointing to the column in
    #   training_co_occurrences corresponding to each context
    # - train_word_ids is a dictionary mapping contexts to numerical indices pointing to the row in
    #   training_co_occurrences corresponding to each word
    # - train_word_freqs is NumPy array with as many positions as there are rows in the co-occurrence matrix; each
    #   position stores the frequency count of the word in the corresponding row of the co-occurrence matrix
    train_co_occurrences, context_ids, train_word_ids, train_word_freqs = collect_contexts(input_corpus, training_perc,
                                                                                           pos_dict, bigrams=bigrams,
                                                                                           trigrams=trigrams)
    # print the co-occurrence matrix to file for later use
    ctxt.create_vector_space(train_co_occurrences, train_word_ids, context_ids, training_space)

    print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished creating the vector space from the training corpus.")
    print()

    print()
    print(strftime("%Y-%m-%d %H:%M:%S") + ": Started computing context statistics from the training corpus.")

    # use TiMBL to compute feature weights for each context, based on the co-occurrence counts derived before
    feature_weights = compute_feature_weights(training_space, weights_file)

    # compute several distributional statistics for each context, and print them to a summary file (specified at the top
    # of this function)
    compute_context_statistics(train_co_occurrences, train_word_freqs, context_ids, feature_weights,
                               context_output, training_perc)

    print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished computing context statistics from the training corpus.")
    print()

    if not leave_one_out:
        # in case the train-test setting was selected (default) this block of the code is executed
        print()
        print(strftime("%Y-%m-%d %H:%M:%S") + ": Started creating the vector space from the test corpus.")

        # collect co-occurrence count in the portion of corpus allotted to the test set, only using the contexts
        # harvested during training: words that don't occur in any of these contexts are lost, and so are contexts that
        # didn't occur during training. The training and test matrix have to be perfectly aligned column-wise to
        # meaningfully compute similarities, hence the features collected at training are used also when processing the
        # test corpus.
        # - test_co_occurrences is analoguous to training_co_occurrences
        # - test_word_ids is analoguous to training_word_ids
        # - test_word_freqs is analoguous to train word_freqs
        test_space = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'test']), str(training_perc), ext]))
        test_co_occurrences, test_word_ids, test_word_freqs = create_test_space(input_corpus, test_perc, context_ids,
                                                                                pos_dict, bigrams=bigrams,
                                                                                trigrams=trigrams)
        ctxt.create_vector_space(test_co_occurrences, test_word_ids, context_ids, test_space)
        target_word_frequencies = test_word_freqs
        target_co_occurrences = test_co_occurrences
        target_word_ids = test_word_ids

        print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished creating the vector space from the test corpus.")
        print()

    else:
        # in case the leave-one-out setting is chosen (by passing True to the 'leave_one_out' parameter, make all
        # structures pertaining to the test corpus None so that they're not considered in further functions. Also, use
        # word frequency counts, co-occurrence matrix, and word_ids derived from the training corpus
        test_word_ids, test_co_occurrences, test_space, context_ids = [None, None, None, None]
        target_word_frequencies = train_word_freqs
        target_co_occurrences = train_co_occurrences
        target_word_ids = train_word_ids

    print()
    print(strftime("%Y-%m-%d %H:%M:%S") + ": Started running the PoS tagging experiment and deriving word statistics.")

    accuracies = defaultdict(dict)
    if sklearn:
        # if sklearn is selected, use cosine as a distance metric and categorize target items (training words in the
        # leave-one-out setting and test words in the train-test setting), then compute statistics for the target words
        # using the appropriate co-occurrence matrix and corresponding row indices
        sklearn_accuracies, sim, word_ids = ctxt.sklearn_cosine_similarity(train_co_occurrences, train_word_ids,
                                                                           test_words=test_word_ids,
                                                                           test_space=test_co_occurrences,
                                                                           contexts=context_ids,
                                                                           diag_value=diag_value, nn=nn,
                                                                           plot=similarities_plot)
        for word in sklearn_accuracies:
            accuracies[word]['correct'] = sklearn_accuracies[word]['correct']
            accuracies[word]['sklearn'] = {'predicted': sklearn_accuracies[word]['predicted'],
                                           'accuracy': sklearn_accuracies[word]['accuracy']}

    if timbl:
        # if timbl is selected, use overlap as a distance metric and categorize target items (training words in the
        # leave-one-out setting and test words in the train-test setting), then compute statistics for the target words
        # using the appropriate co-occurrence matrix and corresponding row indices
        categorization_output = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'posTagged']),
                                                                  str(training_perc), ext]))
        timbl_accuracies = ctxt.timbl_experiment(training_space, categorization_output, nn=nn, test=test_space)
        for word in timbl_accuracies:
            accuracies[word]['correct'] = timbl_accuracies[word]['correct']
            accuracies[word]['timbl'] = {'predicted': timbl_accuracies[word]['predicted'],
                                         'accuracy': timbl_accuracies[word]['accuracy']}

    # too many word_ids wrt to the words categorized in the experiment

    output = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'wordStats']), ext]))
    compute_words_statistics(target_co_occurrences, target_word_ids, target_word_frequencies, accuracies,
                             feature_weights, output, training_perc)

    print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished running the PoS tagging experiment and deriving word statistics.")
    print()


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description="Select distributional contexts and perform "
                                                 "a PoS tagging experiment using TiMBL.")

    parser.add_argument("-c", "--corpus", required=True, dest="corpus",
                        help="Specify the corpus file to be used as input (encoded as a .json file).")
    parser.add_argument("-o", "--output_folder", required=True, dest="output_folder",
                        help="Specify the folder where output files will be stored.")
    parser.add_argument("-p", "--pos_file", required=True, dest="pos_file",
                        help="Specify the path to the file containing the PoS mapping from CHILDES tags to custom"
                             "coarser tags.")
    parser.add_argument("-l", "--leave_one_out", action="store_false", dest="leave_one_out",
                        help="Specify whether to use a leave-one-out setting where words from the training set are"
                             "categorized searching the same set of words (except for the target) for the nearest "
                             "neighbours. Default is true: selecting the option -l causes the program NOT to run"
                             "a leave-one-out experiment.")
    parser.add_argument("-n", "--nearest_neighbours", dest="nn", default=1,
                        help="Specify the number of nearest neighbours to be considered for categorization.")
    parser.add_argument("-d", "--diagonal", dest="diag_value", default=None,
                        help="Specify the value to which the main diagonal of the cosine matrix is set."
                             "Default is None meaning that nothing happens; by setting the value to 0, categorization"
                             "is forced to discard same word types when finding nearest neighbours, ensuring"
                             "generalization.")
    parser.add_argument("-m", "--minimum_training", dest="min_train", default=40,
                        help="Specify the minimum percentage of utterances from each input corpus to be used for "
                             "training (default is 40).")
    parser.add_argument("-s", "--steps", dest="steps", default=7,
                        help="Specify the number of training steps (default is 7, meaning that the model is "
                             "trained first using 40% of the utterances, then 45%, then 50%, and so on, up to 70%,"
                             "resulting in 7 steps).")
    parser.add_argument("-T", "--test_perc", dest="test_perc", default=30,
                        help="Specify the percentage of utterances from each input corpus to be used for testing "
                             "(default is 30). The sum of the value passed to -T and to -M need to add to 100.")
    parser.add_argument("-b", "--bigrams", action="store_true", dest="bigrams",
                        help="Specify whether to consider bigrams as relevant distributional contexts.")
    parser.add_argument("-t", "--trigrams", action="store_true", dest="trigrams",
                        help="Specify whether to consider trigrams as relevant distributional contexts.")
    parser.add_argument("-K", "--sklearn", action="store_true", dest="sklearn",
                        help="Specify whether to use sklearn cosine computation to perform the PoS tagging experiment.")
    parser.add_argument("-B", "--timbl", action="store_true", dest="timbl",
                        help="Specify whether to use TiMBL to perform the PoS tagging experiment.")

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    max_train = 100 if args.leave_one_out else 100 - int(args.test_perc)
    min_train = int(args.min_train)
    steps = int(args.steps)
    test_perc = int(args.test_perc)
    diag_value = args.diag_value if args.diag_value is None else int(args.diag_value)
    pos_dict = ctxt.read_category_mapping(args.pos_file)
    training_steps = np.linspace(min_train, max_train, num=steps)
    for train_perc in training_steps:
        context_analysis(args.corpus, pos_dict, args.output_folder, training_perc=int(train_perc), test_perc=test_perc,
                         bigrams=args.bigrams, trigrams=args.trigrams, diag_value=diag_value,
                         nn=int(args.nn), sklearn=args.sklearn, timbl=args.timbl, leave_one_out=args.leave_one_out)


########################################################################################################################


if __name__ == '__main__':

    main()
