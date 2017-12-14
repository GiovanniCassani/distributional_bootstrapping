import os
from collections import defaultdict


def table2dict(in_file, sep, header=True, c=3, t=50):

    """
    :param in_file: the path to a .txt file containing several columns, where the first one consists of unique items
                    that can be therefore used as keys in a dictionary
    :param sep:     a string indicating the character separating columns in the input file
    :param header:  a boolean specifying whether the input file has column headers that specify column names
    :param c:       the column number (1-indexed) containing the information about the percentage of the corpus analyzed
    :param t:       a scalar indicating at which percentage of the corpus the function stops considering elements
    :return d:      a dictionary where the items in the first column of the input file are used as keys and the items in
                    all other columns are concatenated using the separator character and assigned to the corresponding
                    key as value
    """

    d = defaultdict(dict)
    col_names = {}

    with open(in_file, 'r') as f:

        line = f.readline()
        record = line.strip().split(sep)
        for idx, field in enumerate(record):
            if header:
                col_names[idx] = field
            else:
                col_names[idx] = "col" + str(idx)
                d[record[idx]][col_names[idx]] = record[idx]

        for line in f:
            record = line.strip().split(sep)
            try:
                if int(record[c - 1]) == t:
                    for i in range(len(record)):
                        d[record[0]][col_names[i]] = record[i]
            except IndexError:
                pass

    return d, col_names


########################################################################################################################


def intersect(corpora, wd, file_ext, sep='\t'):

    """
    :param corpora:     an iterable containing strings indicating the corpora's names to be considered
    :param wd:          a string pointing to the directory where files to be accessed are stored
    :param file_ext:    a string indicating how all files to be accessed end, after the corpus name
    :param sep:         the character separating the different fields in the input files
    :return shared:     a set containing the words that are common to all input files

    This function takes summary tables created running the main function in this module, for both contexts and words:
    it reads in the first input file, store everything in a dictionary using words or contexts as keys (depending on the
    value of file_ext). Iteratively, all input files are read in, the words or contexts in them are stored and only the
    items that also appeared in all other corpora analyzed so far are retained. In the end, the function gives the set
    of items (words or contexts) that appear in the summary table of all corpora provided as input.

    The input files should all be stored in the same directory, begin with the corpus name, and end with the same
    string, e.g. '/User/Folder1/Luke_wordStats.txt': '/User/Folder1' is wd, 'Luke' is the corpus name, and
    '_wordStats.txt' is the file_ext, i.e. whatever comes after the corpus name.
    """

    shared = set()
    for corpus in corpora:
        infile = os.path.join(wd, ''.join([corpus, file_ext]))
        d, names = table2dict(infile, sep)
        if not shared:
            shared = set(d.keys())
        else:
            shared = set(d.keys()).intersection(shared)

    return shared


########################################################################################################################


def print_targets(targets, output, corpora, wd, file_ext, sep='\t', header=True):

    """
    :param targets:     an iterable containing strings indicating the items in the input files to be retained
    :param output:      the name of the file where the function will print its output
    :param corpora:     an iterable containing strings indicating the corpora's names to be considered
    :param wd:          a string indicating the directory containing all input files
    :param file_ext:    a string indicating how all files to be accessed end, after the corpus name
    :param sep:         the character separating the different fields in the input files
    :param header:      a boolean specifying whether the input files have a header, which will be printed at the top
                        of the output file
    """

    if not wd.endswith("/"):
        wd += "/"

    flag = 0

    with open(wd + output, "a+") as fo:
        for corpus in corpora:
            infile = wd + corpus + file_ext
            with open(infile, "r") as fi:
                if header and not flag:
                    colnames = fi.readline()
                    fo.write(colnames)
                    flag = 1

                for line in fi:
                    target = line.strip().split(sep)[0]
                    if target in targets:
                        fo.write(line)


########################################################################################################################


def get_experimental_targets(corpora, wd, ext, out):

    targets = intersect(corpora, wd, ext)
    print_targets(targets, out, corpora, wd, ext)
