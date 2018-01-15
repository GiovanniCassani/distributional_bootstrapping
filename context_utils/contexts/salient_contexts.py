__author__ = 'GCassani'

"""Get the most useful distributional contexts from a corpus, based on their salience (a combination of
frequency, lexical diversity, and predictability given the words they co-occur with)"""


import operator
import numpy as np
from time import strftime
from datetime import datetime
from context_utils.readers import read_txt_corpus_file
from context_utils.corpus import get_words_and_contexts
from context_utils.utterance import strip_pos, construct_window, get_ngrams
from context_utils.vector_spaces.maker import sort_items
from context_scores import compute_context_score, get_averages


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


def get_useful_contexts(input_corpus, pos_dict=None, k=1, boundaries=True, bigrams=True, trigrams=True, pred=True,
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
    :param pred:            a boolean indicating whether average predictability of contexts given words are
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
        avg_freq, avg_div, avg_pred = get_averages(co_occurrences, word_frequencies)
    else:
        avg_freq, avg_div, avg_pred = [None, None, None]

    contexts_scores = compute_context_score(co_occurrences, contexts2ids, word_frequencies,
                                            pred=pred, div=div, freq=freq,
                                            avg_pred=avg_pred, avg_freq=avg_freq, avg_lex_div=avg_div)

    # only return contexts whose salience score is higher than the threshold t
    return dict((key, value) for key, value in contexts_scores.items() if value > k)
