__author__ = 'GCassani'

"""Create summary files to store statistics on the outcome of the context selection experiments"""


def make_summary_file(path):

    """
    :param path:    the file where the function prints a header, only if the path points to a non existing file
    """

    with open(path, 'x') as o_f:
        o_f.write('\t'.join(['model', 'boundaries', 'age', 'time', 'corpus', 'training_sentences', 'training_tokens',
                             'training_types', 'test_sentences', 'test_tokens', 'test_types', 'selected',
                             'useless_test', 'missed_test', 'timbl_accuracy', 'sklearn_accuracy']))
        o_f.write('\n')


########################################################################################################################


def make_categorization_file(path):

    """
    :param path:    the file where the function prints a header, only if the path points to a non existing file
    """

    with open(path, 'x') as c_f:
        c_f.write('\t'.join(['model', 'boundaries', 'age', 'time', 'corpus', 'word', 'correct',
                             'timbl_predicted', 'timbl_accuracy', 'sklearn_predicted', 'sklearn_accuracy']))
        c_f.write('\n')
