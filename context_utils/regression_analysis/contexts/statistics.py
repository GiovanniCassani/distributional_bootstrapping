import os
import scipy.stats
import operator
import numpy as np
from collections import defaultdict


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
