import os
from collections import defaultdict


def timbl_experiment(training, output_file, test=None, nn=1):

    """
    :param training:        the path to a tab separated .csv file (created using create_vector_space in this module) to
                            be used as training space
    :param output_file:     the path to the file where the result of the TiMBL experiment is printed
    :param test:            the path to a tab separated .csv file (created using create_vector_space in this module) to
                            be used as test space; the default is None, meaning that a leave-one-out categorization is
                            performed on the training set. If a file is passed, then it is used as test set

    :param nn:              the number of nearest neighbours to consider
    :return accuracies:     a dictionary mapping each word to the classification outcome, 1 if the word was classified
                            correctly, 0 otherwise; each word is also mapped to its correct and predicted PoS tag
    """

    accuracies = defaultdict(dict)

    with open(training, 'r') as t:
        first_line = t.readline()
        n = len(first_line.strip().split("\t")) + 100

    train_space = ' -f ' + training
    test_space = ' -t ' + test if test else ' -t leave_one_out'
    out_file = ' -o ' + output_file
    timbl_cmd = 'Timbl -k' + str(nn) + ' -mN:I1 -w0 -N' + str(n) + train_space + test_space + out_file
    os.system(timbl_cmd)

    with open(output_file, "r") as f:
        for line in f:
            record = line.strip().split()
            word = record[0]
            target = record[-2]
            predicted = record[-1]
            accuracies[word]['accuracy'] = 1 if target == predicted else 0
            accuracies[word]['correct'] = target
            accuracies[word]['predicted'] = predicted

    return accuracies
