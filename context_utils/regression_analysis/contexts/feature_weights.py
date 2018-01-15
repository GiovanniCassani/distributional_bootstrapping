__author__ = 'GCassani'

"""Compute feature weights for the relevant contexts given an input co-occurrence vector space"""


import os
from collections import defaultdict


def compute_feature_weights(training_space, output_file):

    """
    :param training_space:  the path to a file containing the co-occurrence count matrix derived from the training
                            corpus
    :param output_file:     the path to a file where the weight of each context will be written
    :return weights:        a dictionary mapping each context to 4 strings, each indicating one of the possible
                            weighting schemes: gain ratio ('gr'), information gain ('ig'), X-square ('x2'), and shared
                            variance ('sv'). Each string map to the weight of the corresponding contexts under the
                            weighting scheme at hand. All scores are stored for later processing.
    """

    weights = defaultdict(dict)

    with open(training_space, 'r') as t:
        first_line = t.readline()
        n = len(first_line.strip().split("\t")) + 100

    train_space = ' -f ' + training_space
    out_file = ' -W ' + output_file
    timbl_cmd = 'timbl -mN:I1 -N' + str(n) + train_space + out_file
    print(timbl_cmd)
    os.system(timbl_cmd)

    with open(output_file, "r") as f:

        gr, ig, x2, sv = [0, 0, 0, 0]

        for line in f:

            if line.strip() == '# gr':
                gr, ig, x2, sv = [1, 0, 0, 0]
            elif line.strip() == '# ig':
                gr, ig, x2, sv = [0, 1, 0, 0]
            elif line.strip() == '# x2':
                gr, ig, x2, sv = [0, 0, 1, 0]
            elif line.strip() == '# sv':
                gr, ig, x2, sv = [0, 0, 0, 1]

            if any([gr, ig, x2, sv]):
                try:
                    feature, weight = line.strip().split("\t")
                    if gr:
                        weights[int(feature) - 2]['gr'] = float(weight)
                    elif ig:
                        weights[int(feature) - 2]['ig'] = float(weight)
                    elif x2:
                        weights[int(feature) - 2]['x2'] = float(weight)
                    elif sv:
                        weights[int(feature) - 2]['sv'] = float(weight)
                except ValueError:
                    pass

    return weights
