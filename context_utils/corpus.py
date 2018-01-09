from context_utils.utterance import clean_utterance, construct_window, get_ngrams


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


def count_cds_types(corpus, pos_dict=None):

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


def get_words_and_contexts(tokens, filtered_corpus, min_length, size, boundaries=True,
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
