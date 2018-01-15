__author__ = 'GCassani'

"""Process each utterance to extract lexically specific distributional contexts"""


import re
import numpy as np


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

    # create an empty output list consisting of 'NA's. size is doubled because it indicates the number of elements to
    # the left and to the right of the pivot. The pivot is added last. The list is 0 indexed so the index marked by size
    # is the middle one
    window = ['NA'] * (size * 2)
    window.append('NA')
    window[size] = 'X'

    clean_words = []
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
