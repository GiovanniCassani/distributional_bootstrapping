__author__ = 'GCassani'

"""Compute correlation between predictability and average conditional probability of contexts given words"""


import os
import numpy as np
import pandas as pd
from context_utils.contexts.salient_contexts import get_useful_contexts
from context_utils.contexts.context_scores import compute_predictability
from context_utils.vector_spaces.maker import create_vector_space


def pred2condprob(input_corpus, output_folder, pos_dict=None, k=0, boundaries=True, bigrams=True, trigrams=True,
                  pred=True, div=True, freq=True, averages=True):

    """
    :param input_corpus:    the path to a .txt file containing CHILDES transcripts, with one utterance per line and
                            words divided by white spaces. The first element of each utterance is the capitalized label
                            of the speaker, as found in CHILDES. The second element is a dummy word marking the
                            beginning of the utterance, #start; the last element is a dummy word marking the end of the
                            utterance, #end. Each word is paired to its Part-of-Speech tag, the two separated by a
                            tilde, word~PoS.
    :param output_folder:   a string indicating the folder where simulation results will be stored
    :param pos_dict:        a dictionary mapping CHILDES PoS tags to custom tags; any CHILDES tag that doesn't appear as
                            key in the dictionary is assumed to be irrelevant and words labeled with those tags are
                            discarded; the default to None means that every input word is considered
    :param k:               the threshold to determine which contexts are salient: every context whose score is higher
                            than t is considered to be salient. The default is 1.
    :param boundaries:      a boolean indicating whether utterance boundaries are to be considered or not as context
    :param bigrams:         a boolean indicating whether bigrams are to be collected
    :param trigrams:        a boolean indicating whether trigrams are to be collected
    :param pred:            a boolean indicating whether average predictability of contexts given words are
                            a relevant piece of information in deciding about how important a context it
    :param div:             a boolean indicating whether lexical diversity of contexts, i.e. the number of different
                            words that occur in a context, is a relevant piece of information in deciding about how
                            important a context it
    :param freq:            a boolean indicating whether contexts' frequency count is a relevant piece of information in
                            deciding about how important a context it
    :param averages:        a boolean specifying whether frequency, diversity, and predictability values for each
                            context have to be compared to running averages
    :return correlation:    the Pearson correlation between the predictability score and the average conditional
                            probability for each context
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = '/'.join([output_folder, 'pred2condprob.csv'])

    salient_contexts = get_useful_contexts(input_corpus, pos_dict=pos_dict, k=k, boundaries=boundaries,
                                           bigrams=bigrams, trigrams=trigrams, pred=pred, div=-div, freq=freq,
                                           averages=averages)
    co_occurrences, useless, unused, word_ids, context_ids = create_vector_space(input_corpus, salient_contexts,
                                                                                 pos_dict=pos_dict,
                                                                                 boundaries=boundaries,
                                                                                 bigrams=bigrams, trigrams=trigrams)

    word_frequencies = get_word_frequencies(co_occurrences, word_ids)

    df = pd.DataFrame(index=np.arange(0, len(context_ids)),
                      columns=['context', 'predictability', 'avg_cond_prob'])

    logged_co_occurrences = np.zeros(co_occurrences.shape)
    logged_co_occurrences[np.nonzero(co_occurrences)] = np.log10(co_occurrences[np.nonzero(co_occurrences)])

    for context, idx in context_ids.items():

        word_counts = co_occurrences[:, idx]
        logged_word_counts = logged_co_occurrences[:, idx]

        if np.log10(np.count_nonzero(logged_word_counts)):
            predictability = compute_predictability(logged_word_counts, word_frequencies)
            avg_conditional_probability = compute_average_conditional_probability(word_counts, word_frequencies)
        else:
            predictability, avg_conditional_probability = [0, 0]

        df.loc[idx] = [context, predictability, avg_conditional_probability]

    df.to_csv(output_file, sep='\t', index=False)

    df['predictability'] = np.float64(df['predictability'])
    df['avg_cond_prob'] = np.float64(df['avg_cond_prob'])

    correlation = df['predictability'].corr(df['avg_cond_prob'])

    return correlation


########################################################################################################################


def compute_average_conditional_probability(co_occurrences, word_frequencies):

    """
    :param word_frequencies:    a dictionary mapping words to their frequency count in the corpus
    :param co_occurrences:      the column vector corresponding to the current context, containing its co-occurrence
                                counts with all the words in the corpus
    :return:                    the average predictability value for the current context given all the words it
                                co-occurred with in the corpus
    """

    # initialize an empty vector with as many cells as there are non-zero cells in the input vector of co-occurrences
    p = np.zeros(np.count_nonzero(co_occurrences))

    # go through each word that has a non-zero co-occurrence count with the current context, get its frequency, its
    # co-occurrence value, and then compute its predictability; finally get the average predictability for the current
    # context
    for i, row in enumerate(np.nonzero(co_occurrences)[0]):
        word_frequency = word_frequencies[row]
        co_occurrence = co_occurrences[row]
        c = co_occurrence / word_frequency
        if c <= 1:
            p[i] = co_occurrence / word_frequency
        else:
            print('Avg conditional probability larger than 1: %f / %f.') % (co_occurrence, word_frequency)
    avg_conditional_probability = np.mean(p)

    return avg_conditional_probability


########################################################################################################################


def get_word_frequencies(co_occurrences, word_ids):

    """
    :param co_occurrences:      a NumPy 2d array containing co-occurrence counts between words (rows) and distributional
                                contexts (columns)
    :param word_ids:            a dictionary mapping words to their row indices in the co-occurrence matrix
    :return word_frequencies:   a NumPy 1d array mapping words to their frequencies as derived from the co-occurrence
                                matrix (the index in the array is the index in the co-occurrence matrix)
    """

    word_frequencies = np.zeros([len(word_ids)])
    row_frequencies = co_occurrences.sum(1)
    for r_idx, frequency in enumerate(row_frequencies):
        word_frequencies[r_idx] = frequency

    return word_frequencies
