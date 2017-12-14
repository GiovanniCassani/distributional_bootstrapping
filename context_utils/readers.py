import csv


def read_txt_corpus_file(path, delimiter=' '):

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
