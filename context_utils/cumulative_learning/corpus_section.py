__author__ = 'GCassani'

"""Pre-process corpora to use them in the context selection experiment (see cumulative_learning.py)"""

import os
from context_utils.corpus import count_cds_types, count_cds_tokens, count_cds_lines


def make_corpus_section(output_file, ages, i, target_dir='', pos_dict=None, training=True):

    """
    :param output_file:     the path to the file that will be created
    :param ages:            a list of file-names
    :param i:               an index (between 0 and the length of the input list)
    :param target_dir:      the path to the folder where the newly created file will be moved. The default is the empty
                            string, meaning that the file is not moved from the current working directory
    :param pos_dict:        a dictionary mapping CHILDES PoS tags to custom tags; any CHILDES tag that doesn't appear as
                            key in the dictionary is assumed to be irrelevant and words labeled with those tags are
                            discarded; the default to None means that every input word is considered
    :param training:        a boolean specifying whether the function creates the training file or not (in which case it
                            automatically creates a test file). In case a training file is created, all files up to the
                            index i are printed to the output_file
    :return sentences:      the number of sentences from adult speakers in the output file
    :return tokens:         the number of tokens (from the allowed PoS tags, if so desired) in the outputs file
    :return types:          the number of types (from the allowed PoS tags, if so desired) in the output file
    """

    # create the output file according to the input specification
    destination = '/'.join([target_dir, output_file]) if target_dir else output_file
    if not os.path.exists(destination):
        if training:
            cmd = 'cat ' + ' '.join(ages[:i+1]) + ' >> ' + output_file
        else:
            cmd = 'cat ' + ' '.join(ages[i:]) + ' >> ' + output_file
        os.system(cmd)

        if target_dir:
            cmd = ' '.join(['mv', output_file, target_dir])
            os.system(cmd)
    else:
        print("The file at %s already exists." % destination)

    # get counts for sentences, tokens, and types
    sentences = count_cds_lines(destination)
    tokens = count_cds_tokens(destination, pos_dict=pos_dict)
    types = count_cds_types(destination, pos_dict=pos_dict)

    return sentences, tokens, types
