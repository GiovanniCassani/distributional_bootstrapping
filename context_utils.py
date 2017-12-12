__author__ = 'GCassani'


import os
import re
import csv
import operator
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from time import strftime
from collections import defaultdict, Counter
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.metrics.pairwise import cosine_similarity as cos
    import numpy as np


def read_txt_corpusfile(path, delimiter=' '):

    """
    :param path:        the path to a txt file
    :param delimiter:   the character separating elements in each line of the input file; default is blank space
    :return corpus:     a list of lists of strings, where every inner list is a line from the input file
    """

    with open(path, newline='') as f:
        corpus = list(csv.reader(f, delimiter=delimiter))

    return corpus


########################################################################################################################


def read_category_mapping(pos_map):

    """
    :param pos_map:     the path to a file containing the mapping between CHILDES PoS tags and the custom tagset. The
                        file must consist of two white-space-separated columns, each containing a string
    :return pos_dict:   a dictionary mapping the strings from the first column in the input file to the corresponding
                        strings from the second column
    """

    pos_dict = {}

    with open(pos_map, 'r') as r:
        for line in r:
            pos = line.rstrip("\n").split()
            pos_dict[pos[0]] = pos[1]

    return pos_dict


########################################################################################################################


def read_targets(path, pos_dict=None):

    """
    :param path:        the path to a .txt file containing one word per line, consisting of a PoS tag and a word form,
                        separated by a tilde ('~')
    :param pos_dict:    a dictionary mapping CHILDES PoS tags to custom tags
    :return targets:    a set containing the words from the input file: if a pos_dict is passed, the original PoS tags
                        are replaced by the corresponding tags in the dictionary, otherwise the original tags are
                        preserved
    """
    targets = set()
    with open(path, "r") as f:
        for line in f:
            pos, word = line.strip().split('~')
            pos = pos_dict[pos] if pos_dict else pos
            targets.add('~'.join([pos, word]))

    return targets


########################################################################################################################


def count_cds_lines(corpus):

    """
    :param corpus:      the path to a .txt file containing CHILDES transcripts, with one utterance per line and
                        words divided by white spaces. The first element of each utterance is the capitalized
                        label of the speaker, as found in CHILDES. The second element is a dummy word marking
                        the beginning of the utterance, #start; the last element is a dummy word marking the end
                        of the utterance, #end. Each word is paired to its Part-of-Speech tag, the two separated
                        by a tilde, word~PoS.
    :return utterances: an integer indicating how many utterances from the input corpus were not produced by the child
    """

    utterances = 0

    with open(corpus, 'r+') as f:
        for line in f:
            words = line.strip().split(' ')
            if words[0] != 'CHI':
                utterances += 1

    return utterances


########################################################################################################################


def count_cds_unique_words(corpus, pos_dict=None):

    """
    :param corpus:          the path to a .txt file containing CHILDES transcripts, with one utterance per line and
                            words divided by white spaces. The first element of each utterance is the capitalized
                            label of the speaker, as found in CHILDES. The second element is a dummy word marking
                            the beginning of the utterance, #start; the last element is a dummy word marking the end
                            of the utterance, #end. Each word is paired to its Part-of-Speech tag, the two separated
                            by a tilde, word~PoS.
    :param pos_dict:        if a dictionary mapping CHILDES PoS tags to custom tags is passed, any word tagged with
                            with a label that is not a key in the dictionary is not considered. If the default is kept,
                            no words are discarded from processing
    :return types:          an integer indicating how many different types there are in the utterances from the input
                            corpus tagged as not having been produced by the child
    """

    word_set = set()

    with open(corpus, 'r+') as f:
        for line in f:
            tokens = line.strip().split(' ')
            if tokens[0] != 'CHI':
                words = tokens[2:-1]
                for w in words:
                    if pos_dict:
                        if w.split('~')[1] in pos_dict:
                            word_set.add(w)
                    else:
                        word_set.add(w)

    types = len(word_set)
    return types


########################################################################################################################


def count_cds_tokens(corpus, pos_dict=None):

    """
    :param corpus:          the path to a .txt file containing CHILDES transcripts, with one utterance per line and
                            words divided by white spaces. The first element of each utterance is the capitalized
                            label of the speaker, as found in CHILDES. The second element is a dummy word marking
                            the beginning of the utterance, #start; the last element is a dummy word marking the end
                            of the utterance, #end. Each word is paired to its Part-of-Speech tag, the two separated
                            by a tilde, word~PoS.
    :param pos_dict:        if a dictionary mapping CHILDES PoS tags to custom tags is passed, any word tagged with
                            with a label that is not a key in the dictionary is not considered. If the default is kept,
                            no words are discarded from processing
    :return tokens:         an integer indicating how many different tokens there are in the utterances from the input
                            corpus tagged as not having been produced by the child
    """

    tokens = 0

    with open(corpus, 'r+') as f:
        for line in f:
            utterance = line.strip().split(' ')
            clean_words = []
            if utterance[0] != 'CHI':
                words = utterance[2:-1]
                for w in words:
                    if pos_dict:
                        if w.split('~')[1] in pos_dict:
                            clean_words.append(w)
                    else:
                        clean_words.append(w)

                tokens += len(clean_words)

    return tokens


########################################################################################################################


def split_into_selection_train_test(corpus, test_utterances=3000, training_utterances=7000):

    """
    :param corpus:              the path to a .txt file containing CHILDES transcripts, with one utterance per line and
                                words divided by white spaces. The first element of each utterance is the capitalized
                                label of the speaker, as found in CHILDES. The second element is a dummy word marking
                                the beginning of the utterance, #start; the last element is a dummy word marking the end
                                of the utterance, #end. Each word is paired to its Part-of-Speech tag, the two separated
                                by a tilde, word~PoS.
    :param test_utterances:     the number of utterances to be included in the test set (taken from the end of the input
                                corpus)
    :param training_utterances: the number of utterances to be included in the training set (taken from the middle of
                                the input corpus, just before the utterances allocated to the test set)
    """

    cds_counter = 0
    selection_file = os.getcwd()+'/'+corpus.split('.')[0]+'_dev.txt'
    train_file = os.getcwd()+'/'+corpus.split('.')[0]+'_train.txt'
    test_file = os.getcwd()+'/'+corpus.split('.')[0]+'_test.txt'

    tmp_sel_file = os.getcwd()+'/sel.txt'
    tmp_train_file = os.getcwd()+'/train.txt'
    tmp_test_file = os.getcwd()+'/test.txt'

    for line in reversed(open(corpus).readlines()):
        tokens = line.strip().split(' ')
        if tokens[0] != 'CHI':
            cds_counter += 1
        # first print utterances to the temporary file of the right section in reversed order
        if cds_counter <= test_utterances:
            with open(tmp_test_file, 'a+') as tmp_test:
                tmp_test.write(line)
        elif cds_counter <= test_utterances + training_utterances:
            with open(tmp_train_file, 'a+') as tmp_train:
                tmp_train.write(line)
        else:
            with open(tmp_sel_file, 'a+') as tmp_sel:
                tmp_sel.write(line)

    # then reverse the temporary files and print in the correct order to the output files
    for line in reversed(open(tmp_sel_file).readlines()):
        with open(selection_file, 'a+') as f_sel:
            f_sel.write(line)

    for line in reversed(open(tmp_train_file).readlines()):
        with open(train_file, 'a+') as f_train:
            f_train.write(line)

    for line in reversed(open(tmp_test_file).readlines()):
        with open(test_file, 'a+') as f_test:
            f_test.write(line)

    # get rid of the temporary files
    os.remove(tmp_sel_file)
    os.remove(tmp_test_file)
    os.remove(tmp_train_file)


########################################################################################################################


def words_and_contexts(tokens, filtered_corpus, min_length, size, boundaries=True,
                       lemmas=None, pos_dict=None, bigrams=True, trigrams=True):

    """
    :param tokens:          a list of strings
    :param filtered_corpus: a list of lists, containing strings; all utterances containing legitimate words are added to
                            this list, which will contain only the utterances from the corpus that meet the input
                            criteria
    :param min_length:      the minimum number of strings in a clean utterance for it to be considered legitimate
    :param size:            the size of the window around each target word, in which contexts are collected
    :param boundaries:      a boolean indicating whether to consider utterance boundaries as legitimate contexts or not
    :param lemmas:          a list of strings of the same length as tokens; if one is passed, tokens are assumed to be
                            simple strings, not carrying any PoS information, which is taken from the lemmas, which are
                            in turn supposed to consist of a word and a PoS tag, joined by a tilde ('~'); if no lemmas
                            list is passed, tokens are taken to carry PoS tag information, also in the form word~PoS
    :param pos_dict:        a dictionary mapping CHILDES tags to custom ones; default is None, meaning that everything
                            is left unchanged; if a dictionary is passed, all words tagged with tags not in the
                            dictionary are discarded from further processing, and the original tags are replaced with
                            the custom ones
    :param bigrams:         a boolean specifying whether bigrams, i.e. contexts consisting of a lexical item and an
                            empty slot, such as the_X, or X_of, are to be collected
    :param trigrams:        a boolean specifying whether trigrams, i.e. contexts consisting of two lexical items and an
                            empty slot, such as in_the_X, or X_of_you, or the_X_of, are to be collected
    :return words:          a set with the words from the current utterance
    :return contexts:       a set with the contexts from the current utterance
    """

    utterance = clean_utterance(tokens, lemmas=lemmas, pos_dict=pos_dict, boundaries=boundaries)

    words = set()
    contexts = set()
    idx = 1 if boundaries else 0
    last_idx = len(utterance) - 1 if boundaries else len(utterance)

    # if at least one valid word was present in the utterance and survived the filtering stage, collect all possible
    # contexts from the utterance, as specified by the input granularities
    if len(utterance) > min_length:
        filtered_corpus.append(utterance)
        while idx < last_idx:
            # using every word as pivot, collect all contexts around a pivot word and store both contexts and words
            context_window = construct_window(utterance, idx, size)
            current_contexts = get_ngrams(context_window, bigrams=bigrams, trigrams=trigrams)
            words.add(utterance[idx])
            for context in current_contexts:
                contexts.add(context)
            idx += 1

    # return the set of unique words and unique contexts derived from the utterance provided as input
    return words, contexts


########################################################################################################################


def clean_utterance(words, pos_dict=None, lemmas=None, boundaries=True):

    """
    :param words:           a list of strings
    :param pos_dict:        a dictionary mapping CHILDES PoS tags to custom, more coarse tags (if the mapping is from
                            coarse to fine-grained the dictionary doesn't have a many-to-one mapping and thus would have
                            different values mapped to the same key, defying the feasibility of an automatic mapping)
    :param lemmas:          a list containing the same number of elements as words, but containing lemmas and
                            Part-of-Speech tags, encoded as a string like 'dog~N', with word and PoS tag being
                            separated by a tilde ('~'). Default is None: in this case, the function assumes that the
                            PoS tags come with the strings in the first list, always as word~PoS strings
    :param boundaries:      a boolean specifying whether to consider boundary elements, marked by a starting hash '#'
    :return words_clean:    a list of strings containing all strings from the input one that matched the following
                            criteria: their first substring is not empty; their second sub-string is not in the set
                            passed as second argument; are not boundary elements, if the parameter boundaries is set to
                            False
    """

    words_clean = ['#bound~#start'] if boundaries else []

    # handle utterance boundaries according to the input parameter choice
    # get PoS tag from aligned lemmas if present, or from tokens directly if no lemmas are passed
    # get rid of non alphabetic and non numeric characters in the input word, such as hyphens, brackets, and so on...
    # map the PoS tag to the custom set if a mapping is provided, leave untouched otherwise
    # reverse the word~PoS sequence to PoS~word for leater processing
    for w in range(len(words)):
        if not words[w].startswith('~'):
            pos_tag = lemmas[w].split('~')[1] if lemmas else words[w].split('~')[1]
            word = words[w] if lemmas else words[w].split('~')[0]
            word = re.sub(r"[^a-zA-Z0-9']", '', word)
            if pos_dict:
                if pos_tag in pos_dict:
                    new_tag = pos_dict[pos_tag]
                    words_clean.append("~".join([new_tag, word]))
            else:
                words_clean.append("~".join([pos_tag, word]))

    if boundaries:
        words_clean.append('#bound~#end')

    return words_clean


########################################################################################################################


def strip_pos(el, i=0, sep1='~', sep2='__', context=False):

    """
    :param el:          a string, consisting of two or more substrings separated by the same character
    :param i:           the index (0-based) of the substring to be kept; default to 0, meaning that the first substring
                        is preserved
    :param sep1:        the character separating the relevant substrings, default to a tilde ('~')
    :param sep2:        a further separator character that further subdivides component substring into sub-substrings;
                        default to two underscores ('__')
    :param context:     a boolean specifying whether the function needs to parse a distributional context which consists
                        of multiple words, each consisting of a word-form and a PoS tag. The function first splits
                        components of the context, then retains the desired substring from each component, and glues the
                        components back together with the provided separator (argument to sep2)
    :return outcome:    the input string, with PoS information stripped away
    """

    if context:
        outcome = []
        constituents = el.split(sep2)
        for constituent in constituents:
            try:
                outcome.append(strip_pos(constituent, sep1=sep1, i=i))
            except IndexError:
                outcome.append(constituent)
        outcome = sep2.join(outcome)
    else:
        outcome = el.split(sep1)[i]

    return outcome


########################################################################################################################


def construct_window(words, i, size, splitter=''):

    """
    :param words:       a list of strings (non-empty)
    :param i:           an integer indicating an index larger than 0 and smaller than or equal to the length of the
                        input list
    :param size:        an integer specifying how large the window around the target index should be to the left
                        and to the right of the middle element.
    :param splitter:    an optional argument indicating the character that divides PoS tags from words in the input
                        input list
    :return window:     a list of uneven length, determined by the value of size, where words[i] is the central element
                        and to its right and left there are the other elements in the input list, in their respective
                        positions. If the value of size is larger than the number of elements next to target item in the
                        input list, the output list is padded with 'NA's

    """

    clean_words = []
    # create an empty output list consisting of 'NA's. size is doubled because it indicates the number of elements to
    # the left and to the right of the pivot. The pivot is added last.
    window = ['NA'] * (size * 2)
    window.append('NA')
    # the list is 0 indexed so the index marked by size is the middle one
    window[size] = 'X'

    if splitter:
        for word in words:
            clean_words.append(word.split(splitter)[1])
    else:
        clean_words = words

    for j in range(1, size + 1):

        # idx1_l is the index in the window being created; idx2_l is the index in the input list
        # idx1_l goes to the left of the middle element of the new window being created
        # idx2_l is responsible of grabbing words to the left of the target words, if there are there are
        idx1_l = size - j
        idx2_l = i - j

        # if the second index identifies an item in the input list, the position in the window being created identified
        # by idx1_l is replaced with the list item identified by idx2_l
        if idx2_l >= 0:
            window[idx1_l] = clean_words[idx2_l]

        # the same mechanism is applied to the right side of the window being created, checking that there are words in
        # the input list at the position indicated by idx2_r and substituting the position in the window being created
        # identified by the new idx1_r with the item from the input list identified by idx2_r
        idx1_r = size + j
        idx2_r = i + j
        if idx2_r < len(clean_words):
            window[idx1_r] = clean_words[idx2_r]

    return window


########################################################################################################################


def get_ngrams(w, bigrams=True, trigrams=True, sep='__'):

    """
    :param w:           a list, obtained with the function construct_window
    :param bigrams:     a boolean indicating whether bigrams are to be considered
    :param trigrams:    a boolean indicating whether trigrams are to be considered
    :param sep:         a string indicating how constituents of n-grams are glued together
    :return n_grams:    a list containing all the relevant contexts created from the input list according to the
                        specified parameter values
    """

    n_grams = list()
    t = int(np.floor(len(w)/2))
    avoid = {'NA'}

    if bigrams:
        # collect all bigrams that don't contain NAs
        if w[t - 1] not in avoid:
            n_grams.append(sep.join([w[t - 1], w[t]]))
        if w[t + 1] not in avoid:
            n_grams.append(sep.join([w[t], w[t + 1]]))

    if trigrams:
        # collect all trigrams that don't contain NAs
        if w[t - 1] not in avoid and w[t + 1] not in avoid:
            n_grams.append(sep.join([w[t - 1], w[t], w[t + 1]]))
        if t >= 2:
            if w[t + 2] not in avoid:
                n_grams.append(sep.join([w[t], w[t + 1], w[t + 2]]))
            if w[t - 2] not in avoid:
                n_grams.append(sep.join([w[t - 2], w[t - 1], w[t]]))

    return n_grams


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


def diversity_cutoff(np_array, k, ids, rows=True):

    """
    :param np_array:    a 2d NumPy array
    :param k:           the cutoff number: only columns pointing to contexts that occurred at least twice with
                        at least k words are considered
    :param ids:         a dictionary mapping strings to integers
    :param rows:        a boolean indicating whether to consider rows (default) or columns
    :return outcome:    a set containing the strings whose lexical diversity along the specified dimension is 1
    """

    outcome = set()
    k = int(k)
    # count how many cells along rows or columns (depending on the value of the parameter row) have non-zero values
    lexical_diversities = (np_array > 0).sum(1) if rows else (np_array > 0).sum(0)

    # keep the dimensions that had enough non-zero cells, where enough is specified by the value assigned to the
    # k parameter
    target_ids = np.nonzero(lexical_diversities >= k)[0]

    reversed_ids = {v: k for k, v in ids.items()}
    for i in target_ids:
        outcome.add(reversed_ids[i])

    return outcome


########################################################################################################################


def create_vector_space(co_occurrence, row_ids, col_ids, output_file):

    """
    :param co_occurrence:   a NumPy 2d array, where each row is a word and each column is a context. Each cell contains
                            an integer specifying how many times a word and a context co-occurred in the input corpus
    :param row_ids:         a dictionary mapping strings denoting words to their row index in the co-occurrence matrix
    :param col_ids:         a dictionary mapping strings denoting contexts to their column index in the co-occurrence
                            matrix
    :param output_file:     the path to the file where the co-occurrence matrix will be written
    """

    with open(output_file, 'a+') as f:
        for word, r_idx in sorted(row_ids.items(), key=operator.itemgetter(1)):
            counts = []
            pos, w = word.split('~')
            if w != '':
                for context, c_idx in sorted(col_ids.items(), key=operator.itemgetter(1)):
                    counts.append(co_occurrence[r_idx, c_idx])
                f.write("\t".join([word, "\t".join(["\t".join([str(c) for c in counts]), pos])]))
                f.write('\n')


########################################################################################################################


def get_salient_frames(frames, n):

    """
    :param frames:  a dictionary mapping strings to integers
    :param n:       the number of elements from the input dictionary to be returned
    :return:        a dictionary mapping strings to integers, containing the top n elements (sorted by value in
                    descending order) from the input dictionary
    """

    output_frames = {}
    c = 1
    for k in sorted(frames.items(), key=operator.itemgetter(1), reverse=True):
        if c <= n:
            output_frames[k[0]] = frames[k[0]]
            c += 1
        else:
            return output_frames


########################################################################################################################


def make_matrix(input_matrix, row_target_indices, row_source_indices, col_indices):

    """
    :param input_matrix:        a NumPy array where columns are contexts (strings corresponding to each index ought to
                                be provided in the argument col_indices) and rows are words (strings corresponding to
                                each index ought to be provided in the argument row_source_indices
    :param row_target_indices:  a dictionary mapping word strings to numerical indices, to be used as row indices in the
                                oputput matrix
    :param row_source_indices:  a dictionary mapping word strings to numerical indices, indicating to which string each
                                row in the input matrix corresponds
    :param col_indices:         a dictionary mapping context strings to numerical indices, indicating column indices in
                                the input matrix and in the target matrix
    :return output_matrix:      a NumPy array with as many rows as there are entries in the row_target_indices and as
                                many columns as there are contexts, with each cell containing the word-context
                                co-occurrence count from the input matrix
    """

    output_matrix = np.zeros((len(row_target_indices), len(col_indices)))

    # loop through all words from the input matrix and contexts, getting the corresponding row and column indices. Then
    # get the co-occurrence count for the word-context pair. If there isn't such a pair (because the word is missing,
    # the context is missing, or the two never occurred together in the input corpus), set the corresponding
    # cell in the output matrix to 0
    for w in row_source_indices:
        for k in col_indices:
            r_source = row_source_indices[w]
            r_target = row_target_indices[w]
            c = col_indices[k]
            try:
                output_matrix[r_target, c] = input_matrix[r_source, c]
            except KeyError:
                output_matrix[r_target, c] = 0

    return output_matrix


########################################################################################################################


def plot_matrix(x, output_path='', neighbors=0):

    """
    :param x:           a 2d NumPy array
    :param output_path: the path to the file where the plot will be saved
    :param neighbors:   the number of nearest neighbour to be shown in the matrix plot. Default to 0 means that all 
                        neighbours are displayed; setting a different, positive number means that only the neighbours
                        up to the desired number are gonna be printed with all the other cells artificially set to 0
                        so not to clutter the visualization.
    """

    r, c = x.shape
    fig, ax_array = plt.subplots(1, 1, figsize=(10, 10), sharex='col', sharey='row')

    for ax in np.ravel(ax_array):

        if neighbors:
            # if only the nearest neighbors have to be plotted, first independently sort all columns in the array of
            x_sorted = np.sort(x, axis=0)[::-1]

            # initialize a vector of the same length, loop through all columns, turn them into a set to avoid counting
            # separately all rows sharing the same value, sort them in descending order, and get the value at the
            # desired index putting it into the newly initialized vector. If the vector of values doesn't contain enough
            # values for the index to actually point to a value, take the last value in the vector. The resulting vector
            # contains the value of the desired index for all columns (or the largest possible one).
            threshold = np.zeros(c)
            for col in range(c):
                try:
                    threshold[col] = sorted(set(x_sorted[:, col]), reverse=True)[neighbors]
                except IndexError:
                    threshold[col] = sorted(set(x_sorted[:, col]), reverse=True)[-1]

            # finally set to 0 all cosine similarity values in the NumPy array that are strictly smaller than the
            # threshold value in the corresponding column and plot
            x[np.where(x < threshold)] = 0
            ax.imshow(x, aspect='auto', interpolation='nearest')

        else:
            ax.imshow(x, aspect='auto', interpolation='nearest')

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()


########################################################################################################################


def timbl_experiment(training, output_file, test=None, nn=1):

    """
    :param training:        the path to a tab separated .csv file (created using create_vector_space in this module) to
                            be used as training space
    :param output_file:     the path to the file where the result of the TiMBL experiment is printed
    :param test:            the path to a tab separated .csv file (created using create_vector_space in this module) to
                            be used as test space; the default is None, meaning that a leave-one-out categorization is
                            performed on the training set. If a file is passed, then it is used as test set

    :param nn:              the number of nearest neighbours to consider
    :return accuracies:     a dictionary mapping each word to the classification outcome, 1 if the word was classified
                            correctly, 0 otherwise; each word is also mapped to its correct and predicted PoS tag
    """

    accuracies = defaultdict(dict)

    with open(training, 'r') as t:
        first_line = t.readline()
        n = len(first_line.strip().split("\t")) + 100

    train_space = ' -f ' + training
    test_space = ' -t ' + test if test else ' -t leave_one_out'
    out_file = ' -o ' + output_file
    timbl_cmd = 'timbl -k' + str(nn) + ' -mN:I1 -w0 -N' + str(n) + train_space + test_space + out_file
    os.system(timbl_cmd)

    with open(output_file, "r") as f:
        for line in f:
            record = line.strip().split()
            word = record[0]
            target = record[-2]
            predicted = record[-1]
            accuracies[word]['accuracy'] = 1 if target == predicted else 0
            accuracies[word]['correct'] = target
            accuracies[word]['predicted'] = predicted

    return accuracies


########################################################################################################################


def sklearn_cosine_similarity(training_space, training_words,
                              test_space=None, test_words=None, contexts=None,
                              pos_mapping=None, nn=1, diag_value=None, plot=''):

    """
    :param training_space:  a 2d NumPy array storing word-context co-occurrence counts derived from the training corpus
    :param training_words:  a dictionary mapping words from the training space to the corresponding row indices in the
                            training space
    :param test_space:      a 2d NumPy array storing word-context co-occurrence counts derived from the test corpus
    :param test_words:      a dictionary mapping words from the test space to the corresponding row indices in the
                            test space. If a test space is passed, test_words has to be assigned a value, otherwise the
                            function will throw an error
    :param contexts:        a dictionary mapping contexts to their column indices in the training and test spaces;
                            default is None, because this mapping is only used in the train-test setting to keep the
                            alignment between training and test spaces
    :param pos_mapping:     a dictionary mapping CHILDES PoS tags to custom, coarser tags
    :param nn:              the number of nearest neighbours to be considered when categorizing a test word
    :param diag_value:      the value to which all the cells on the main diagonal of the matrix of cosine similarities
                            between test and training vectors are set (default is 0, meaning that cells on the main
                            diagonal don't impact the nearest neighbour computation). This option makes it possible to
                            force the model to categorize a test word while ignoring the vector from the training space
                            that correspond to the same word type, thus enforcing generalization
    :param plot:            a string indicating the path where the plot showing the cosine similarity matrix is saved
                            The default is the empty string, meaning that no plot is created
    :return hits:           a dictionary mapping each word in the test set to three fields and the corresponding value:
                            'predicted' is the PoS tag that the learner predicted for a test word
                            'correct' is the correct PoS tag as found in the CHILDES corpus
                            'accuracy' is a binary value indicating if 'predicted' and 'correct' match (1) or not (0)
    """

    t = 1 if test_space is not None else 0
    w = 1 if test_words is not None else 0
    c = 1 if contexts is not None else 0
    if sum([t, w, c]) not in [0, 3]:
        raise ValueError('Unsure whether to use a leave-one-out or training-test approach! '
                         'If you want to run a leave-one-out experiment, do not provide any argument to the parameters'
                         ' test_space, test_words, and contexts. If, however, you want to perform an experiment in the'
                         ' training-test setting, provide appropriate arguments to all three parameters.')

    hits = defaultdict(dict)

    if test_space is not None:
        # use a training-test setting, where words from the test set are categorized by retrieving nearest neighbours in
        # the training set
        target_words = test_words
        words = set(training_words.keys()).union(set(test_words.keys()))

        # map every word occurring in either the training space, the test space, or both to a numerical index and get
        # an inverted mapping from indices to strings
        word_indices = sort_items(words)
        inverted_word_indices = {v: k for k, v in word_indices.items()}

        # create a training matrix and a test matrix that have as many rows as there are words in total, and the same
        # columns as the original matrices; then compute pairwise similarities between each pair of training-test words
        training_space = make_matrix(training_space, word_indices, training_words, contexts)
        test_space = make_matrix(test_space, word_indices, test_words, contexts)
        cosine_similarities = cos(training_space, test_space)

        # if so specified in the function call, set the diagonal values to the desired number
        # the idea is to 'silence' the diagonal by setting it to 0: this because the diagonal cells correspond to the
        # cosine similarity between equal types in the training and test set (e.g. dog in the training set and dog in
        # the test set). The cosine will not be 1 because the vectors of co-occurrence will differ (they have been
        # harvested in two different corpora); yet, we can expect same types to have more similar co-occurrence patterns
        # then different types. This could bias the retrieval of nearest neighbours: dog (from the training set) will be
        # retrieved as nearest neighbour of dog (from the test set). This is not a problem per se, but it can be in some
        # experimental settings: the diag-Value allows to get rid of this by force the diagonal values to 0, so that no
        # same word from training word will be retrieved as nearest neighbour for any test item
        if diag_value is not None:
            cosine_similarities[np.diag_indices_from(cosine_similarities)] = diag_value

    else:
        # use a leave-one-out setting, where words from the training set are categorized by retrieving nearest
        # neighbours from the training set, excluding the vector of the word being categorized from the pool of possible
        # neighbours
        target_words = training_words
        words = training_words
        word_indices = sort_items(words)
        inverted_word_indices = {v: k for k, v in word_indices.items()}
        cosine_similarities = cos(training_space)

        # in a leave-one-out setting, the diagonal is always set to 0 because otherwise categorization would be perfect:
        # the same vectors would be compared, resulting in a cosine similarity of 1, which will always be the maximum.
        # To avoid this, the diagonal cells are forced to 0.
        cosine_similarities[np.diag_indices_from(cosine_similarities)] = 0

    if plot:
        plot_matrix(cosine_similarities, neighbors=10, output_path=plot)

    # Use the derived cosine similarities to find which words from the training set are closer to each of the target
    # words (which words are used as targets depend on whether a test space is passed: if it is, target words are test
    # words, if it's not, target words are training words) to be able to categorize the target words. Nearest neighbors
    # are retrieved using a nearest distance approach, meaning that when two or more words from the training set are at
    # the same closest distance from a target word, they are all considered as nearest neighbors to assign a PoS tag to
    # the target word. Ties are broken by looking for the most frequent neighbour in the training set. If there is a tie
    # a word is sammpled randomly from the pool of most frequent words among the neighbours.
    for word in target_words:
        # get the column index of the test word to be categorized, and get the indices of all the rows that have a
        # cosine similarity to the word to be categorized that is at least as high as the closest distance (if k is 1,
        # otherwise get the cosine similarity value corresponding to the second closest distance (k=2), third closest
        # distance (k=3), and so on)
        c_idx = word_indices[word]
        nearest_indices = get_nearest_indices(cosine_similarities, c_idx, nn=nn)

        # get all the word strings having a high enough cosine similarity value to the word to be categorized
        nearest_neighbors = get_nearest_neighbors(nearest_indices[0], inverted_word_indices)

        # if more than one neighbour is found at the closest distance, pick the one with the highest frequency of
        # occurrence in the training set; if more than a word has the same frequency count, pick randomly
        predicted = categorize(nearest_neighbors, training_space,
                               word_indices, pos_mapping=pos_mapping)
        hits[word]['predicted'] = predicted
        hits[word]['correct'] = pos_mapping[word.split('~')[0]] if pos_mapping else word.split('~')[0]
        hits[word]['accuracy'] = 1 if hits[word]['predicted'] == hits[word]['correct'] else 0

    return hits, cosine_similarities, word_indices


########################################################################################################################


def get_nearest_indices(cosine_similarities, idx, nn=1):

    """
    :param cosine_similarities: a NumPy 2-dimensional array
    :param idx:                 an integer indicating which column to consider
    :param nn:                  an integer indicating the number of nearest neighbours to consider (the function uses
                                nearest distances rather than neighbours: if two or more words are at the same closest
                                distance they're all consider - when nn=1, as in the default)
    :return nearest_indices:    a tuple whose first element contains the row indices from the input NumPy array
                                indicating the cells with the highest values in the column indicated by the input
                                parameter idx. The second element of the tuple is empty
    """

    # sort all the columns in the NumPy array independently and in descending order
    cosine_similarities_sorted = np.sort(cosine_similarities, axis=0)[::-1]

    # get the value corresponding to the closest distance (if nn=1) or second closest distance (if nn=2), and so on
    # if the vector is shorter then the chosen value for nn, the function simply takes the smallest value in the column,
    # which is the last one since the column is sorted in descending order
    try:
        t = sorted(set(cosine_similarities_sorted[:, idx]), reverse=True)[nn-1]
    except IndexError:
        t = sorted(set(cosine_similarities_sorted[:, idx]), reverse=True)[-1]

    # get the vector of row indices from the original, unsorted NumPy array that have a distance equal or higher than
    # the value of the desired number of neighbours (distances) set by nn
    nearest_indices = np.where(cosine_similarities[:, idx] >= t)

    return nearest_indices


########################################################################################################################


def get_nearest_neighbors(nearest_indices, words):

    """
    :param nearest_indices: a tuple whose first element contains the row indices from the input NumPy array indicating
                            the cells with the highest values in the column indicated by the input parameter idx. The
                            second element of the tuple is empty.
    :param words:           a dictionary mapping numerical indices to word strings
    :return neighbors:      a set of strings containing those strings that match the indices in the input tuple
    """

    neighbors = set()
    for i in nearest_indices:
        neighbors.add(words[i])

    return list(neighbors)


########################################################################################################################


def tally_tags(l, pos_mapping=None):

    """
    :param l:               an iterable of strings, consisting of a word form and a PoS tag separated by a tilde ("~")
    :param pos_mapping:     a dictionary mapping PoS tags to more coarse labels. Default is None, meaning that the PoS
                            tags found in the strings are considered. If a dictionary is passed, each PoS tag found in a
                            string is mapped to the corresponding label
    :return tallied_tags:   an sorted list of tuples, each containing a string as first element (a PoS tag) and a
                            frequency count as second element, indicating the frequency count of the PoS tag among the
                            nearest neighbors provided in the input iterable
    """

    pos_tags = list()
    for i in l:
        # isolate the PoS tag
        tag = i.split("~")[0]

        # map it to the corresponding label if a mapping is provided or store it as it is
        if pos_mapping:
            pos_tags.append(pos_mapping[tag])
        else:
            pos_tags.append(tag)

    # count frequencies of occurrence for each tag in the list of neighbours and return the resulting list of tuples
    tallied_tags = Counter(pos_tags).most_common()
    return tallied_tags


########################################################################################################################


def categorize(nearest_neighbours, training_matrix, word_indices, pos_mapping=None):

    """
    :param nearest_neighbours:  a list of strings, each consisting of a PoS tag and a word form separated by a tilde
    :param training_matrix:     a 2-dimensional NumPy array containing co-occurrence counts
    :param word_indices:        a dictionary mapping strings to indices to be used as row indices in accessing the NumPy
                                array
    :param pos_mapping:         a dictionary mapping PoS tags to more coarse labels. Default is None, meaning that the
                                PoS tags found in the strings are considered. If a dictionary is passed, each PoS tag
                                found in a string is mapped to the corresponding label
    :return predicted:          a string indicating the predicted PoS tag given the tallied tags and the nearest
                                neighbours together with the frequency information contained in the training matrix
    """

    # Resolve ties by picking the PoS tag of the nearest neighbour that occurred more frequently in the training set, by
    # looking at the co-occurrence pattern: occurrences of the words outside of relevant contexts are not considered;
    # if frequency is enough to break the tie, pick randomly a tag from the most frequent words
    if len(nearest_neighbours) == 1:
        predicted = pos_mapping[nearest_neighbours[0].split('~')[0]] if pos_mapping else \
            nearest_neighbours[0].split('~')[0]
    else:
        max_freq = 0
        most_frequent = []
        for neighbour in nearest_neighbours:
            r_idx = word_indices[neighbour]
            freq = sum(training_matrix[r_idx, :])
            pos = pos_mapping[neighbour.split("~")[0]] if pos_mapping else neighbour.split("~")[0]
            if freq > max_freq:
                max_freq = freq
                most_frequent = [pos]
            elif freq == max_freq:
                most_frequent.append(pos)

        if len(most_frequent) > 1:
            i = int(np.random.randint(0, high=len(most_frequent), size=1))
            predicted = most_frequent[i]
        else:
            predicted = most_frequent[0]

    return predicted


########################################################################################################################


def precision_and_recall(hits, category):

    """
    :param hits:        a dictionary of dictionaries mapping a word to three fields:
                        hits[word]['predicted'] gives the PoS tag predicted by the model for the word
                        hits[word]['correct'] gives the correct PoS tag for the word
                        hits[word]['accuracy'] tells whether predicted and correct PoS tags match (1) or not (0)
    :param category:    a string indicating the PoS being considered: only words whose predicted or correct PoS tag are
                        considered to compute statistics (to get global statistics for all the categories of interest,
                        simply include this function in a loop over all the categories of interest)
    :return tp:         the number of true positives
    :return tp_fp:      the sum of true and false positives
    :return tp_fn:      the sum of true positives and false negatives
    """

    tp = 0
    tp_fp = 0
    tp_fn = 0

    for k in hits:
        if hits[k]['predicted'] == category:
            tp_fp += 1

        if hits[k]['correct'] == category:
            tp_fn += 1

        if hits[k]['predicted'] == category and hits[k]['correct'] == category:
            tp += 1

    return tp, tp_fp, tp_fn


########################################################################################################################


def sort_items(l):

    """
    :param l:           an iterable containing strings
    :return indices:    a dictionary mapping the same strings to integers indicating the numerical index corresponding
                        to each word. Keys are sorted before generating the indices.
    """

    indices = {}
    idx = 0

    for k in sorted(l):
        # map strings to integers, to make look-up by string easier
        indices[k] = idx
        idx += 1

    return indices


########################################################################################################################


def model_id(cond_prob=True, div=True, freq=True, bigrams=True, trigrams=True):

    """
    :param cond_prob:       a boolean indicating whether average conditional probabilities of contexts given words are
                            a relevant piece of information in deciding about how important a context it
    :param div:             a boolean indicating whether lexical diversity of contexts, i.e. the number of different
                            words that occur in a context, is a relevant piece of information in deciding about how
                            important a context it
    :param freq:            a boolean indicating whether contexts' frequency count is a relevant piece of information in
                            deciding about how important a context it
    :param bigrams:         a boolean indicating whether bigrams are to be collected
    :param trigrams:        a boolean indicating whether trigrams are to be collected
    :return:                a string encoding which input options were true: it uniquely identifies a model in the
                            possible parameter space
    """

    info = ''
    info = ''.join([info, 'c']) if cond_prob else info
    info = ''.join([info, 'd']) if div else info
    info = ''.join([info, 'f']) if freq else info

    context = ''
    context = ''.join([context, 'b']) if bigrams else context
    context = ''.join([context, 't']) if trigrams else context

    return '_'.join([info, context])


def harvest_frames(input_corpus, pos_dict=None, boundaries=True, freq_frames=True, flex_frames=False):

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
    :param boundaries:      a boolean indicating whether utterance boundaries are to be considered or not as context
                            elements.
    :param freq_frames:     a boolean indicating whether frequent frames are collected
    :param flex_frames:     a boolean indicating whether flexible frames are collected
    :return frames:         a dictionary containing the n most frequent frames as keys and the corresponding frequency
                            counts as values; words within a frame are separated by two underscores ('__')

    The function makes it possible to harvest frequent and flexible frames at the same time, but the two are stored in
    the same dictionary: if both parameters are set to True, a way must be devised to discriminate frequent and flexible
    frames afterwards: it's fairly trivial given the structure of the frames, but it's not implemented here. Ideally,
    this function is called with either of frequent and flexible frames, but not both.
    """

    frames = Counter()
    corpus = read_txt_corpusfile(input_corpus)

    sep = '__'

    for line in corpus:
        if line[0] != 'CHI':
            # get rid of the speaker ID once made sure the utterance doesn't come from the child
            del line[0]

            # get rid of unwanted elements from the original utterance
            words = clean_utterance(line, boundaries=boundaries, pos_dict=pos_dict)

            # get the numerical index of the last element in the utterance. If frequent frames are to be collected,
            # subtract one from the last index and add one to the first index, because no trigram can be collected
            # for the first and last elements
            if flex_frames:
                last_idx = len(words)
                idx = 0
                while idx < last_idx:
                    # collect flexible frames, minding that no right context is collected for the end of utterance
                    # dummy word and no left context is collected for the beginning of utterance dummy word; store
                    # the frequency count of each frame
                    if words[idx].split('~', 1)[1] != '#end':
                        frame = sep.join([words[idx].split('~', 1)[1], 'X'])
                        frames[frame] += 1
                    if words[idx].split('~', 1)[1] != '#start':
                        frame = sep.join(['X', words[idx].split('~', 1)[1]])
                        frames[frame] += 1
                    idx += 1

            if freq_frames:
                last_idx = len(words) - 1
                idx = 1
                while idx < last_idx:
                    # collect flexible frames
                    frame = sep.join([words[idx-1].split('~', 1)[1],
                                      'X', words[idx+1].split('~', 1)[1]])
                    frames[frame] += 1

                    idx += 1

    return frames


########################################################################################################################


def learning_contexts(input_corpus, pos_dict=None, k=1, boundaries=True, bigrams=True, trigrams=True, cond_prob=True,
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
    corpus = read_txt_corpusfile(input_corpus)
    filtered_corpus = []
    words = set()
    contexts = set()

    # collect all words and contexts from the corpus, getting rid of PoS tags so that homographs are not disambiguated
    # if they are tagged differently
    for line in corpus:
        # get rid of all utterances uttered by the child and clean child-directed utterances
        if line[0] != 'CHI':
            del line[0]
            w, c = words_and_contexts(line, filtered_corpus, min_length, size, boundaries=boundaries,
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


def build_experiment_vecspace(input_corpus, salient_contexts, pos_dict=None, targets='', to_ignore='',
                              boundaries=True, bigrams=True, trigrams=True):

    """
    :param input_corpus:        the path to a .txt file containing CHILDES transcripts, with one utterance per line and
                                words divided by white spaces. The first element of each utterance is the capitalized
                                label of the speaker, as found in CHILDES. The second element is a dummy word marking
                                the beginning of the utterance, #start; the last element is a dummy word marking the end
                                of the utterance, #end. Each word is paired to its Part-of-Speech tag, the two separated
                                by a tilde, word~PoS.
    :param salient_contexts:    a set containing the contexts determined to be salient
    :param pos_dict:            a dictionary mapping CHILDES PoS tags to custom tags; any CHILDES tag that doesn't
                                appear as key in the dictionary is assumed to be irrelevant and words labeled with those
                                tags are discarded; the default to None means that every word is considered
    :param targets:             the path to a .txt file containing the target words, i.e. the words for which
                                co-occurrence counts will be collected. The file contains one word per line, with words
                                joined to the corresponding PoS tag by a tilde. By default, no file is passed and the
                                function considers all words as targets.
    :param to_ignore:           the path to a .txt file containing the words to be ignored. The file contains one word
                                per line, with words joined to the corresponding PoS tag by a tilde. By default, no file
                                is passed and the function considers all words, without ignoring any.
    :param boundaries:          a boolean indicating whether utterance boundaries are to be considered or not as context
    :param bigrams:             a boolean indicating whether bigrams are to be collected
    :param trigrams:            a boolean indicating whether trigrams are to be collected
    :return co_occurrences:     a dictionary of dictionaries, mapping words to context to the word-context co-occurrence
                                count in the input corpus
    :return useless_contexts:   a set containing the contexts from the input dictionary that never occurred in the
                                corpus or that only occurred with one word (either because they only occur once or
                                because they occur multiple times but always with the same word type)
    :return unused_words:       a set containing the words in the input corpus that occur at least once in at least one
                                of the contexts provided in the input dictionary contexts
    """

    # read in the sets of words to be used as targets and/or those to be avoided, if provided
    target_words = read_targets(targets, pos_dict=pos_dict) if targets else set()
    skip_words = read_targets(to_ignore, pos_dict=pos_dict) if to_ignore else set()

    # set the size of the window around the target word where contexts are collected
    size = 2 if trigrams else 1

    # set the minimum length of a legitimate utterance: 0 if utterance boundaries are not considered and 2 if they are,
    # since there will always at least the two boundary markers
    min_length = 2 if boundaries else 0

    # read in the corpus and initialize a list where to store utterances that survived the cleaning step
    # (i.e. utterances that contained at least one legitimate word that is not a boundary marker, if they are considered
    corpus = read_txt_corpusfile(input_corpus)
    filtered_corpus = []
    words = set()

    # collect all words from the corpus, preserving the PoS tags, since we want to be able to tell whether we categorize
    # each homograph correctly
    for line in corpus:
        # get rid of all utterances uttered by the child and clean child-directed utterances
        if line[0] != 'CHI':
            del line[0]
            w, c = words_and_contexts(line, filtered_corpus, min_length, size,
                                      pos_dict=pos_dict, bigrams=bigrams, trigrams=trigrams)
            words = words.union(w)

    # get the target words for which co-occurrence counts needs to be collected, depending on the set of target words or
    # words to be ignored passed to the function
    targets = set()
    for w in words:
        if target_words:
            if w in target_words and w not in skip_words:
                targets.add(w)
        else:
            if w not in skip_words:
                targets.add(w)

    # map words and contexts (provided in the input) to numerical indices
    targets2ids = sort_items(targets)
    contexts2ids = sort_items(salient_contexts)
    print(strftime("%Y-%m-%d %H:%M:%S") + ": I collected all words in the corpus.")
    print()

    """
    At this point we have two dictionaries:
     - one contains all the words collected in the corpus passed as input to this function and filtered according to the 
       words in the set of target words and in the set of words to be ignored; surviving words are sorted according to
       the PoS tag and then according to the word form, stored as keys in the form PoStag~word and mapped to numerical 
       indices.
     - one contains the contexts passed as input to the function (the contexts that were deemed salient by the 
       function learning_contexts), from which all information about PoS tags has been stripped away; these contexts
       are sorted and mapped to numerical indices
    The numerical indices will point to rows (words) and columns (contexts) of a 2d NumPy array that will store 
    word-context co-occurrence counts. Contexts are sorted because columns have to be aligned across training and test
    spaces. Words are sorted so that words from the same PoS tags are in neighbouring rows and make the visualization
    of further steps easier to grasp
    """

    total_utterances = len(filtered_corpus)
    check_points = {np.floor(total_utterances / float(100) * n): n for n in np.linspace(5, 100, 20)}
    co_occurrences = np.zeros([len(targets2ids), len(salient_contexts)]).astype(float)
    line_idx = 0

    for utterance in filtered_corpus:
        idx = 1 if boundaries else 0
        last_idx = len(utterance) - 1 if boundaries else len(utterance)
        while idx < last_idx:
            current_word = utterance[idx]
            if current_word in targets2ids:
                # only process the word if it is among the targets (i.e. either it occurs in the set of target words or
                # it doesn't occur in the set of words to be ignored, as determined previously)
                w = construct_window(utterance, idx, size, splitter='~')
                curr_contexts = get_ngrams(w, bigrams=bigrams, trigrams=trigrams)
                row_id = targets2ids[current_word]
                for context in curr_contexts:
                    if context in salient_contexts:
                        # only keep track of co-occurrences between target words and salient contexts
                        col_id = contexts2ids[context]
                        co_occurrences[row_id, col_id] += 1
            # move on through the sentence being processed
            idx += 1

        line_idx += 1
        if line_idx in check_points:
            print('Line ' + str(line_idx) + ' has been processed at ' + str(datetime.now()) + '.')

    # get the contexts with lexical diversity lower than 2 (thus salient contexts that never occurred in the input
    # corpus or contexts that only occurred with one word, being useless to any categorization task)
    # the 2 in the function call is the minimum lexical diversity of a context to be considered useful
    # the rows=False argument indicates that the function has to work over columns
    # it returns a set of strings containing the contexts that don't meet the criterion of minimum lexical diversity
    useless_contexts = diversity_cutoff(co_occurrences, 2, contexts2ids, rows=False)

    # create a vector of booleans, with as many values as there are rows in the co-occurrence matrix: this vector has
    # Trues on indices corresponding to rows in the co-occurrence matrix with more than 1 non-zero cell and Falses
    # everywhere else. This vector is used to get rid of 'empty' lines from the co-occurrence matrix and identify
    # indices corresponding to words that never occurred with any of the salient contexts, and then the words
    # corresponding to these indices. Re-align the word-index mapping to the new matrix, taking advantage of the
    # alphabetical order of the indices.
    mask = (co_occurrences > 0).sum(1) > 0
    unused_indices = np.where(mask == False)[0]
    co_occurrences = co_occurrences[mask, :]
    clean_targets2ids = {}
    unused_words = set()

    # loop through the word-index pairs from the smallest index to the largest; if the index is among the unused ones,
    # add the corresponding word to the set of unused word; otherwise assign it a new progressive index that will match
    # the row of the new co-occurrence matrix (this works because the order of the retained rows in the matrix is
    # preserved, so lines at the top of the original matrix will also be at the top of the cleaned matrix and so on)
    new_idx = 0
    for w, i in sorted(targets2ids.items(), key=operator.itemgetter(1)):
        if i in unused_indices:
            unused_words.add(w)
        else:
            clean_targets2ids[w] = new_idx
            new_idx += 1

    return co_occurrences, useless_contexts, unused_words, clean_targets2ids, contexts2ids


########################################################################################################################


def category_f1(hits):

    """
    :param hits:        a dictionary of dictionaries mapping strings (the word types to be categorized) to three fields:
                        'predicted', indicating the PoS tag predicted by the model
                        'correct' indicating the gold standard PoS tag
                        'accuracy', indicating whether the two match (1) or not (0)
    :return stats:      a dictionary of dictionaries mapping each category to its corresponding recall, precision, and
                        f1 scores. Each category is a key of the dictionary and a dictionary itself, whose keys are
                        'recall', 'precision', and 'f1', mapping to the corresponding values. A further key of the
                        top-level dictionary is 'all', consisting of the same three sub-keys, mapping to recall,
                        precision, and f1 scores for the whole experiment.
    """

    stats = defaultdict(dict)

    categories = set()
    for item in hits:
        categories.add(hits[item]['predicted'])
        categories.add(hits[item]['correct'])

    all_tp = []
    all_tp_fn = []
    all_tp_fp = []

    for category in sorted(categories):
        tp, tp_fp, tp_fn = precision_and_recall(hits, category)
        recall = tp / tp_fn if tp_fn != 0 else 0
        precision = tp / tp_fp if tp_fp != 0 else 0
        f1 = 0 if (precision == 0 or recall == 0) else 2 * ((precision * recall) / (precision + recall))

        stats[category]['recall'] = recall
        stats[category]['precision'] = precision
        stats[category]['f1'] = f1

        all_tp.append(tp)
        all_tp_fn.append(tp_fn)
        all_tp_fp.append(tp_fp)

    total_recall = sum(all_tp) / sum(all_tp_fn)
    total_precision = sum(all_tp) / sum(all_tp_fp)
    total_f1 = 2 * ((total_precision * total_recall) / (total_precision + total_recall))

    stats['all']['recall'] = total_recall
    stats['all']['precision'] = total_precision
    stats['all']['f1'] = total_f1

    return stats
