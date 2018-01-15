__author__ = 'GCassani'

"""Process the outcome of the PoS tagging experiment"""


def update_categorization_dict(source_dict, unused_words, target_dict, experiment, model, boundaries,
                               age, time, corpus, pos_mapping=None):

    """
    :param source_dict:     the dictionary containing results of the categorization experiment: each key is a word
                            mapping to the correct PoS tag ('correct'), the predicted PoS tag ('predicted') and the
                            categorization accuracy ('accuracy'), 1 if predicted and correct match, 0 otherwise
    :param unused_words:    a set of words that didn't occur with any of the contexts the model being considered deemed
                            salient; if a word doesn't occur with any context, it's impossible to categorize
    :param target_dict:     a dictionary storing the categorization information for all experiments: a word maps to its
                            correct PoS tag and to a dictionary indicating each experiment that was run. The experiment
                            dictionary consists of two further keys, 'predicted' and 'accuracy'. As an example, the word
                            dog in this dictionary would look something like this:
                            'dog':  'correct':  N
                                    'timbl':    'predicted':    N
                                                'accuracy':     1
                                    'sklearn':  'predicted':    A
                                                'accuracy'      0
                            It is a noun ('correct': N), the timbl experiment predicted noun ('timbl':'predicted':N)
                            and it was correct ('timbl':'accuracy':1) while the sklearn experiment predicted adjective
                            ('sklearn':'predicted':A) and it was wrong ('sklearn':'accuracy':0)
    :param experiment:      the string identifying the experiment being considered
    :param model:           the string indicating the model being considered
    :param boundaries:      a string indicating whether utterance boundaries have been considered or not
    :param age:             the age of the child whose transcript were used to perform the experiment
    :param time:            the time index corresponding to the age
    :param corpus:          the name of the corpus being used
    :param pos_mapping:     a dictionary mapping CHILDES PoS tags to custom ones
    :return:                the target_dict argument, with all words from the source_dict added
    """

    for word in source_dict:
        word_key = '\t'.join([model, boundaries, age, str(time), corpus, word])
        target_dict[word_key]['correct'] = source_dict[word]['correct']
        target_dict[word_key][experiment] = {'predicted': source_dict[word]['predicted'],
                                             'accuracy': source_dict[word]['accuracy']}
    for word in unused_words:
        word_key = '\t'.join([model, boundaries, age, str(time), corpus, word])
        target_dict[word_key]['correct'] = pos_mapping[word.split("~")[0]] if pos_mapping else word.split("~")[0]
        target_dict[word_key][experiment] = {'predicted': 'null',
                                             'accuracy': 0}

    return target_dict


########################################################################################################################


def get_accuracy(d, key1, key2):

    """
    :param d:       a dictionary of dictionaries
    :param key1:    a string indicating a possible first-level key of the dictionary
    :param key2:    a string indicating a possible second-level key of the the first-level key (which is itself a
                    dictionary)
    :return:        the value indicated by the two keys in the input dictionary if one was retrieved, nan otherwise
    """

    try:
        return d[key1][key2]
    except KeyError:
        return float('nan')


########################################################################################################################


def get_predictions(d, key1, key2):

    """
    :param d:       a dictionary of dictionaries
    :param key1:    a string indicating a possible first-level key of the dictionary
    :param key2:    a string indicating a possible second-level key of the the first-level key (which is itself a
                    dictionary)
    :return:        the predicted PoS tag under the categorization experiment indicated by key2, and the accuracy score
                    (1 if the prediction was correct, 0 otherwise); if key2 is missing, a dash is returned as the
                    predicted category and nan as the accuracy
    """

    try:
        return d[key1][key2]['predicted'], d[key1][key2]['accuracy']
    except KeyError:
        return '-', float('nan')
