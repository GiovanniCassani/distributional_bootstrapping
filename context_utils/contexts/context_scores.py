import numpy as np
from collections import Counter


def compute_context_score(co_occurrences, context_ids, word_frequencies,
                          pred=True, div=True, freq=True,
                          avg_freq=None, avg_lex_div=None, avg_pred=None):

    """
    :param co_occurrences:          a 2d NumPy array with rows indicating words and columns indicating contexts; the
                                    cells contain word-context co-occurrence counts
    :param context_ids:             a dictionary mapping contexts to their column indices in the co-occurrence matrix
    :param word_frequencies:        a 1d NumPy array containing frequency counts for words
    :param pred:                    a boolean specifying whether predictability of contexts given words is a relevant
                                    piece of information to compute contexts' salience
    :param div:                     a boolean indicating whether lexical diversity of contexts, i.e. the number of
                                    different words that occur in a context, is a relevant piece of information in
                                    deciding about how important a context it
    :param freq:                    a boolean indicating whether contexts' frequency count is a relevant piece of
                                    information in deciding about how important a context it
    :param avg_freq:                a number indicating the average frequency of all contexts in memory. The default is
                                    None, indicating that the frequency count of each context is considered as is,
                                    without comparing it to a running average
    :param avg_lex_div:             a number indicating the average lexical diversity of all contexts in memory. The
                                    default is None, indicating that the lexical diversity of each context is
                                    considered as is, without comparing it to a running average
    :param avg_pred:                a number indicating the average predictability of contexts given words computed over
                                    all contexts encountered so far; the default is None, indicating that the average
                                    predictability of a context is considered as is, without comparing it to a running
                                    average
    :return context_scores:         a dictionary mapping contexts to their salience score
    """

    # log all the nonzero elements in the co_occurrence matrix: this will turn every cell containing a 1 into a 0
    tmp_co_occurrences = np.zeros(co_occurrences.shape)
    tmp_co_occurrences[np.nonzero(co_occurrences)] = np.log10(co_occurrences[np.nonzero(co_occurrences)])
    context_scores = Counter()

    for context in context_ids:
        # get the column index in the co-occurrence matrix for the current context, and the row indices of all
        # the words that have co-occurred with it at least twice in the corpus; finally, create a vector of the same
        # length as the number of non-zero co-occurrence counts to store predictability scores of the current
        # context given each of the word it co-occurred with at least twice
        col_id = context_ids[context]
        word_counts = tmp_co_occurrences[:, col_id]
        if np.count_nonzero(word_counts):

            # get the frequency count and lexical diversity of the context being considered
            frequency = np.log10(sum(co_occurrences[:, col_id]))
            lexical_diversity = np.log10(np.count_nonzero(word_counts))

            if lexical_diversity:
                predictability = compute_predictability(word_counts, word_frequencies)

                # get the ratios for diversity, frequency and predictability if so required, or simply store the values
                # as such
                d = lexical_diversity / avg_lex_div if avg_lex_div else lexical_diversity
                f = frequency / avg_freq if avg_freq else frequency
                p = predictability / avg_pred if avg_pred else predictability

                # set each variables' value to 1 if the corresponding input parameter is set to False since 1 is the
                # identity value in multiplication and doesn't change the result
                p = p if pred else 1
                f = f if freq else 1
                d = d if div else 1
                context_scores[context] = p * f * d

            else:
                # if the context only occurred at least twice with one word, the logged diversity will be 0, so the
                # result of the multiplication will also be 0, with no need to compute everything
                context_scores[context] = 0

        else:
            # if the context never occurred more than once with any word, the frequency, diversity, and predictability
            # for the context will all be 0, with no need to compute everything
            context_scores[context] = 0

    return context_scores


########################################################################################################################


def get_averages(co_occurrences, word_frequencies):

    """
    :param co_occurrences:      a 2d NumPy array where rows are words, columns are contexts, and cells indicate the
                                word-context co-occurrence count
    :param word_frequencies:    a dictionary mapping words to their frequency count; the frequency count for words
                                cannot be reliably estimated from the co-occurrence matrix because one occurrence of a
                                word is counted with all the contexts it co-occurs with, which can be as many as five if
                                both bigrams and trigrams are considered and the word is not located at an utterance
                                boundary. The frequency count of a word would become disproportionately large if it was
                                computed from the matrix.
    :return avg:                the average length of the sets from the input dictionary, once the length has been
                                logged and all items mapped to sets of cardinality 1 have been discarded (since the log
                                would be 0)
    """

    # sum column-wise co-occurrence counts, then log the frequency counts: co-occurrences of value 1 are considered in
    # the sum, but if the context only occurred once, it is canceled out by the log, which makes contexts with frequency
    # count 1 become 0: these aren't considered when computing the average
    logged_frequencies = np.log10(sum(co_occurrences))
    avg_f = np.mean(logged_frequencies[np.nonzero(logged_frequencies)]) if np.count_nonzero(logged_frequencies) else 1

    # make a copy of the co-occurrence matrix and use that to compute averages so that any in-place modification
    # made while computing averages doesn't cascade onto later processing; then log all the nonzero elements in the
    # co_occurrence matrix to turn every cell containing a 1 into a 0; finally get the non-zero cells column-wise
    tmp_co_occurrences = np.zeros(co_occurrences.shape)
    tmp_co_occurrences[np.nonzero(co_occurrences)] = np.log10(co_occurrences[np.nonzero(co_occurrences)])
    lexical_diversities = (tmp_co_occurrences > 0).sum(0)

    # if there are columns with more than one non-zero co-occurrence value, get the non-zero values, take the log, get
    # rid of columns with diversity 1 (whose log is 0), and compute the average lexical diversity over the remaining
    # columns, i.e. those that in the input matrix have at least 2 non-zero co-occurrence
    if np.count_nonzero(lexical_diversities):
        nonzero_diversities = lexical_diversities[np.nonzero(lexical_diversities)]
        logged_diversities = np.log10(nonzero_diversities)
        avg_l = np.mean(logged_diversities[np.nonzero(logged_diversities)])

        # get the indices of the columns with at least 1 non-zero cell, (contexts that co-occurred more than once with
        # at least one word - the co-occurrence matrix is logged, so all co-occurrence counts originally equal to 1 have
        # become 0, the assumption being that co-occurring once is not informative per se); initialize a vector with as
        # many cells as there are such columns, in order to store the avg predictability for each of the target contexts
        target_columns = np.nonzero(lexical_diversities > 0)[0]
        predictability_scores = np.zeros(len(target_columns))

        # loop through the target contexts, and compute the average predictability for each context over all the words
        # it co-occurred with
        for i, col in enumerate(target_columns):
            word_counts = tmp_co_occurrences[:, col]
            predictability_scores[i] = compute_predictability(word_counts, word_frequencies)

        # average the average predictability socres of all target contexts to get the running average
        avg_p = np.mean(predictability_scores)

    else:
        avg_l, avg_p = [1, 1]

    return avg_f, avg_l, avg_p


########################################################################################################################


def compute_predictability(co_occurrences, word_frequencies):

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
        word_frequency = np.log10(word_frequencies[row])
        co_occurrence = co_occurrences[row]
        c = co_occurrence / word_frequency
        if c <= 1:
            p[i] = co_occurrence / word_frequency
        else:
            print('Predictability larger than 1: %f / %f.') % (co_occurrence, word_frequency)
    predictability = np.mean(p)

    return predictability
