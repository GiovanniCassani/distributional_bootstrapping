__author__ = 'GCassani'

"""Create a unique identifier of the kind of model is being used to perform context selection"""


def make_model_id(pred=True, div=True, freq=True, bigrams=True, trigrams=True):

    """
    :param pred:            a boolean indicating whether average predictability scores of contexts given words are
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
    info = ''.join([info, 'c']) if pred else info
    info = ''.join([info, 'd']) if div else info
    info = ''.join([info, 'f']) if freq else info

    context = ''
    context = ''.join([context, 'b']) if bigrams else context
    context = ''.join([context, 't']) if trigrams else context

    return '_'.join([info, context])