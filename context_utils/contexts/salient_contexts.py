import operator
import numpy as np
from time import strftime
from datetime import datetime
from collections import Counter
from context_utils.readers import read_txt_corpus_file
from context_utils.corpus import get_words_and_contexts
from context_utils.utterance import strip_pos, construct_window, get_ngrams
from context_utils.vector_spaces.maker import sort_items


def print_contexts(contexts, path):

    """
    :param contexts:    a dictionary mapping strings to numbers
    :param path:        the path to the file where contexts are gonna be printed
    """

    with open(path, 'w+') as f:
        for k in sorted(contexts.items(), key=operator.itemgetter(1), reverse=True):
            f.write('\t'.join([k[0], str(k[1])]))
            f.write('\n')


########################################################################################################################


def get_useful_contexts(input_corpus, pos_dict=None, k=1, boundaries=True, bigrams=True, trigrams=True, cond_prob=True,
                        div=True, freq=True, averages=True):

    """
    :param input_corpus:    the path to a .txt file containing CHILDES transcripts, with one utterance per line and
                            words divided by white spaces. The first element of each utterance is the capitalized label
                            of the speaker, as found in CHILDES. The second element is a dummy word marking the
                            beginning of the utterance, #start; the last element is a dummy word marking the end of the
                            utterance, #end. Each word is paired to its Part-of-Speech tag, the two separated by a
                            tilde, word~PoS.
    :param pos_dict:        a dictionary mapping CHILDES PoS tags to custom tags; any CHILDES tag that doesn't appear as
                            key in the dictionary is assumed to be irrelevant and words labeled with those tags are
                            discarded; the default to None means that every input word is considered
    :param k:               the threshold to determine which contexts are salient: every context whose score is higher
                            than t is considered to be salient. The default is 1.
    :param boundaries:      a boolean indicating whether utterance boundaries are to be considered or not as context
    :param bigrams:         a boolean indicating whether bigrams are to be collected
    :param trigrams:        a boolean indicating whether trigrams are to be collected
    :param cond_prob:       a boolean indicating whether average conditional probabilities of contexts given words are
                            a relevant piece of information in deciding about how important a context it
    :param div:             a boolean indicating whether lexical diversity of contexts, i.e. the number of different
                            words that occur in a context, is a relevant piece of information in deciding about how
                            important a context it
    :param freq:            a boolean indicating whether contexts' frequency count is a relevant piece of information in
                            deciding about how important a context it
    :param averages:        a boolean specifying whether frequency, diversity, and predictability values for each
                            context have to be compared to running averages
    :return contexts:       a dictionary mapping contexts to their relevance score
    """

    # set the size of the window around the target word where contexts are collected
    size = 2 if trigrams else 1

    # set the minimum length of a legitimate utterance: 0 if utterance boundaries are not considered and 2 if they are,
    # since there will always at least the two boundary markers
    min_length = 2 if boundaries else 0

    # read in the corpus and initialize a list where to store utterances that survived the cleaning step
    # (i.e. utterances that contained at least one legitimate word that is not a boundary marker, if they are considered
    corpus = read_txt_corpus_file(input_corpus)
    filtered_corpus = []
    words = set()
    contexts = set()

    # collect all words and contexts from the corpus, getting rid of PoS tags so that homographs are not disambiguated
    # if they are tagged differently
    for line in corpus:
        # get rid of all utterances uttered by the child and clean child-directed utterances
        if line[0] != 'CHI':
            del line[0]
            w, c = get_words_and_contexts(line, filtered_corpus, min_length, size, boundaries=boundaries,
                                          pos_dict=pos_dict, bigrams=bigrams, trigrams=trigrams)
            for el in w:
                words.add(strip_pos(el, i=1))
            for el in c:
                contexts.add(strip_pos(el, i=1, context=True))

    # map words and contexts to numerical indices
    words2ids = sort_items(words)
    contexts2ids = sort_items(contexts)
    print(strftime("%Y-%m-%d %H:%M:%S") + ": I collected all words and contexts in the input corpus.")
    print()

    total_utterances = len(filtered_corpus)
    check_points = {np.floor(total_utterances / float(100) * n): n for n in np.linspace(5, 100, 20)}

    # initialize an empty matrix with as many rows as there are words and as many columns as there are contexts in the
    # input corpus, making sure cells store float and not integers
    co_occurrences = np.zeros([len(words2ids), len(contexts2ids)]).astype(float)
    word_frequencies = np.zeros([len(words2ids)])

    line_idx = 0
    for utterance in filtered_corpus:
        # set the first and last index of the utterance, depending on whether utterance boundaries are to be considered
        idx = 1 if boundaries else 0
        last_idx = len(utterance) - 1 if boundaries else len(utterance)
        while idx < last_idx:
            current_word = utterance[idx]
            # collect all contexts for the current pivot word
            context_window = construct_window(utterance, idx, size, splitter='~')
            current_contexts = get_ngrams(context_window, bigrams=bigrams, trigrams=trigrams)
            row_id = words2ids[current_word.split('~')[1]]
            word_frequencies[row_id] += 1
            for context in current_contexts:
                # store the co-occurrence count between the word and context being considered and update
                # their salience score
                col_id = contexts2ids[context]
                co_occurrences[row_id, col_id] += 1
            idx += 1

        # at every 5% of the input corpus, print progress and store summary statistics: nothing is done with them, but
        # a plot can be made, or the values returned
        line_idx += 1
        if line_idx in check_points:
            print('Line ' + str(line_idx) + ' has been processed at ' + str(datetime.now()) + '.')

    if averages:
        avg_freq, avg_div, avg_prob = get_averages(co_occurrences, word_frequencies)
    else:
        avg_freq, avg_div, avg_prob = [None, None, None]

    contexts_scores = compute_context_score(co_occurrences, contexts2ids, word_frequencies,
                                            cond_prob=cond_prob, div=div, freq=freq,
                                            avg_cond_prob=avg_prob, avg_freq=avg_freq, avg_lex_div=avg_div)

    # only return contexts whose salience score is higher than the threshold t
    return dict((key, value) for key, value in contexts_scores.items() if value > k)
    
    
########################################################################################################################


def compute_context_score(co_occurrences, context_ids, word_frequencies,
                          cond_prob=True, div=True, freq=True,
                          avg_freq=None, avg_lex_div=None, avg_cond_prob=None):

    """
    :param co_occurrences:          a 2d NumPy array with rows indicating words and columns indicating contexts; the
                                    cells contain word-context co-occurrence counts
    :param context_ids:             a dictionary mapping contexts to their column indices in the co-occurrence matrix
    :param word_frequencies:        a 1d NumPy array containing frequency counts for words
    :param cond_prob:               a boolean specifying whether conditional probabilities of contexts given words are
                                    a relevant piece of information to compute contexts' salience
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
    :param avg_cond_prob:           a number indicating the average conditional probability of contexts fiven words
                                    over all contexts encountered so far; the default is None, indicating that the
                                    average conditional probability of a context is considered as is, without comparing
                                    it to a running average
    :return context_scores:         a dictionary mapping contexts to their salience score
    """

    # log all the nonzero elements in the co_occurrence matrix: this will turn every cell containing a 1 into a 0
    tmp_co_occurrences = np.zeros(co_occurrences.shape)
    tmp_co_occurrences[np.nonzero(co_occurrences)] = np.log10(co_occurrences[np.nonzero(co_occurrences)])
    context_scores = Counter()

    for context in context_ids:
        # get the column index in the co-occurrence matrix for the context being considered, and the row indices of all
        # the words that have co-occurred with the context at least twice in the corpus (hence, their logged
        # co-occurrence count is larger than 0); finally, initialize a vector of the same length as the number of
        # non-zero co-occurrence counts to store conditional probabilities of the context at hand given each of the
        # word it co-occurred with at least twice
        col_id = context_ids[context]
        word_counts = tmp_co_occurrences[:, col_id]
        if np.count_nonzero(word_counts):
            cp = np.zeros(np.count_nonzero(word_counts))

            # get the frequency count and lexical diversity of the context being considered
            frequency = np.log10(sum(co_occurrences[:, col_id]))
            lexical_diversity = np.log10(np.count_nonzero(word_counts))

            if lexical_diversity:
                # get the average conditional probability of the context at hand given the word it co-occurred with
                for i, row in enumerate(np.nonzero(word_counts)[0]):
                    word_frequency = np.log10(word_frequencies[row])
                    co_occurrence = tmp_co_occurrences[row, col_id]
                    c = co_occurrence / word_frequency
                    if c <= 1:
                        cp[i] = co_occurrence / word_frequency
                    else:
                        print('Conditional probability larger than 1: %f / %f.') % (co_occurrence, word_frequency)
                conditional_probability = np.mean(cp)

                # get the lexical diversity ratio if so required, or simply store the value as such
                ld = lexical_diversity / avg_lex_div if avg_lex_div else lexical_diversity

                # get the frequency ratio if so required, or simply store the value as such
                fr = frequency / avg_freq if avg_freq else frequency

                # get the conditional probability ratio if so required, or simply store the value as such
                cp = conditional_probability / avg_cond_prob if avg_cond_prob else conditional_probability

                # set variables' value to 1 if the corresponding input parameter is set to False since 1 is the
                # identity value in multiplication and doesn't change the result
                c = cp if cond_prob else 1
                f = fr if freq else 1
                d = ld if div else 1
                context_scores[context] = c * f * d

            else:
                # if the context only occurred at least twice with one word, the logged diversity will be 0, so the
                # result of the multiplication will also be 0, with no need to compute everything
                context_scores[context] = 0

        else:
            # if the context never occurred more than once with any word, the frequency, diversity, and conditional
            # probability for the context will all be 0, with no need to compute everything
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
    avg_fr = np.mean(logged_frequencies[np.nonzero(logged_frequencies)]) if np.count_nonzero(logged_frequencies) else 1

    # make a copy of the co-occurrence matrix and use that to compute averages so that any in-place modification
    # made while computing averages doesn't cascade to later processing, introducing hidden differences between
    # the processing of models with or without the computation of averages
    tmp_co_occurrences = np.zeros(co_occurrences.shape)

    # log all the nonzero elements in the co_occurrence matrix: this will turn every cell containing a 1 into a 0
    tmp_co_occurrences[np.nonzero(co_occurrences)] = np.log10(co_occurrences[np.nonzero(co_occurrences)])

    # get the number of non-zero cells column-wise
    lexical_diversities = (tmp_co_occurrences > 0).sum(0)

    # initialize averages for lexical diversity and conditional probability to 1
    avg_ld = 1
    avg_cp = 1

    # if there are columns with more than one non-zero co-occurrence value, get the non-zero values, take the log, get
    # rid of columns with diversity 1 (whose log is 0), and compute the average lexical diversity over the remaining
    # columns, i.e. those that in the input matrix have at least 2 non-zero co-occurrence
    if np.count_nonzero(lexical_diversities):
        nonzero_lds = lexical_diversities[np.nonzero(lexical_diversities)]
        logged_lds = np.log10(nonzero_lds)
        avg_ld = np.mean(logged_lds[np.nonzero(logged_lds)])

        # get the indices of the columns with at least 1 non-zero cell, i.e. the contexts that co-occurred more than
        # once with at least one word - remember that the co-occurrence matrix is logged, so all co-occurrence counts
        # originally equal to 1 have become zero, the assumption being that co-occurring once is not informative per se;
        # initialize a vector with as many positions as there are such columns to store the mean conditional probability
        # for each of the target contexts (i.e. those that co-occurred more than once with at least a word
        target_columns = np.nonzero(lexical_diversities > 0)[0]
        cond_probs = np.zeros(len(target_columns))

        # loop through the target contexts, get the vector of co-occurrences, count the non-zero elements to initialize
        # a NumPy array to store the individual conditional probability for each word that co-occurred with the target
        # context at hand at least twice)
        for i, col in enumerate(target_columns):
            word_counts = tmp_co_occurrences[:, col]
            cp = np.zeros(np.count_nonzero(word_counts))

            # for each word that has a non-zero co-occurrence count with the target context at hand, get the frequency
            # of the word and the co-occurrence count: the counts are logged, so any non-zero cell means the
            # corresponding word and context occurred at least twice alone and together (can be more, of course)
            for j, row in enumerate(np.nonzero(word_counts)[0]):
                word_frequency = np.log10(word_frequencies[row])
                co_occurrence = tmp_co_occurrences[row, col]
                cond_prob = co_occurrence / word_frequency
                if cond_prob <= 1:
                    cp[j] = cond_prob
                else:
                    print('Conditional probability larger than 1: %f / %f.') % (co_occurrence, word_frequency)

            # given that only words that co-occurred at least twice with a target context were considered, we know that
            # the words themselves also occurred at least twice, making it possible to compute all conditional
            # probabilities: now average them and add the result to the vector of conditional probabilities for all
            # target contexts
            cond_probs[i] = np.mean(cp)

        # average the average conditional probabilities of all target contexts to get the running average
        avg_cp = np.mean(cond_probs)

    return avg_fr, avg_ld, avg_cp
