__author__ = 'GCassani'

"""Compute distributional information concerning words given a co-occurrence vector space"""

import os
import scipy.stats
import numpy as np
from collections import defaultdict


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
