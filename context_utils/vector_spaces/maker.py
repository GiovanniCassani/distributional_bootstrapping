__author__ = 'GCassani'

"""Create co-occurrence vector spaces given a corpus and the important distributional contexts"""


import operator
import numpy as np
from time import strftime
from datetime import datetime
from context_utils.readers import read_targets, read_txt_corpus_file
from context_utils.corpus import get_words_and_contexts
from context_utils.utterance import construct_window, get_ngrams


def create_vector_space(input_corpus, salient_contexts, pos_dict=None, targets='', to_ignore='',
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
    corpus = read_txt_corpus_file(input_corpus)
    filtered_corpus = []
    words = set()

    # collect all words from the corpus, preserving the PoS tags, since we want to be able to tell whether we categorize
    # each homograph correctly
    for line in corpus:
        # get rid of all utterances uttered by the child and clean child-directed utterances
        if line[0] != 'CHI':
            del line[0]
            w, c = get_words_and_contexts(line, filtered_corpus, min_length, size,
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

    # create a vector of booleans, with as many values as there are rows in the co-occurrence matrix: this vector is
    # True on indices corresponding to rows in the co-occurrence matrix with more than 1 non-zero cell and False
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


def diversity_cutoff(np_array, k, ids, rows=True):

    """
    :param np_array:    a 2d NumPy array
    :param k:           the cutoff number: only columns pointing to contexts that occurred with at least k words are
                        considered
    :param ids:         a dictionary mapping strings to integers
    :param rows:        a boolean indicating whether to consider rows (default) or columns
    :return outcome:    a set containing the strings corresponding to the indices (along the specified dimension) that
                        don't match the requirement
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
