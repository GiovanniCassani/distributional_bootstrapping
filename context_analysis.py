__author__ = 'GCassani'

import os
import argparse
import numpy as np
from time import strftime
from collections import defaultdict
from context_utils.pos_tagging.timbl import timbl_experiment
from context_utils.pos_tagging.sklearn import sklearn_experiment
from context_utils.readers import read_category_mapping
from context_utils.vector_spaces.printer import print_vector_space
from context_utils.regression_analysis.contexts.contexts import collect_contexts
from context_utils.regression_analysis.contexts.statistics import compute_context_statistics
from context_utils.regression_analysis.contexts.feature_weights import compute_feature_weights
from context_utils.regression_analysis.words.co_occurrences import create_test_space
from context_utils.regression_analysis.words.statistics import compute_words_statistics


def context_analysis(input_corpus, pos_dict, output_dir, leave_one_out=False, training_perc=70, test_perc=30,
                     bigrams=True, trigrams=True, nn=1, diag_value=None, sklearn=True, timbl=False):

    """
    :param input_corpus:    a .json file consisting of a list of lists: each utterance is a list of two lists, the first
                            consists of tokens, the second of corresponding lemmas and PoS tags
    :param pos_dict:        a dictionary mapping CHILDES PoS tags to custom tags
    :param output_dir:      the path of the directory where all the results of the simulation will be printed
    :param leave_one_out:   a boolean to specify whether to use a leave-one-out setting where words from the training
                            set are categorized searching the same set of words (except for the target) for the nearest
                            neighbours. Default is false, meaning that a train-test approach is used
    :param training_perc:   the percentage of the input corpus to be used for training
    :param test_perc:       the percentage of the input corpus to be used for testing (if theleave_one_out parameter is
                            set to False)
    :param bigrams:         a boolean specifying whether to consider bigrams or not
    :param trigrams:        a boolean specifying whether to consider trigrams or not
    :param nn:              the number of nearest neighbours to consider when categorizing a new item
    :param diag_value:      the value of the cells on the main diagonal in the matrix of similarities constructed to
                            retrieve nearest neighbours: default to None means that the diagonal is left untouched.
                            However, setting it to 0 excludes those cells from the possible nearest neighbours: cells on
                            the diagonal contain the similarity between the vectors of the same word type, one collected
                            in the training corpus, the other in the test corpus. Even though the two vectors are not
                            identical, it can still be expected that they will be more similar than vectors for
                            different types. Thus, one can artificially exclude same types from the pool of nearest
                            neighbours. This is done automatically in the leave-one-out setting, since the vectors would
                            be identical, being collected in the same corpus.
    :param sklearn:         a boolean specifying whether to perform the categorization experiment using the sklearn
                            implementation that relies on cosine as a distance metric
    :param timbl:           a boolean specifying whether to perform the categorization experiment using the TiMBL
                            implementation that relies on numeric overlap as a distance metric
    """

    if sum([sklearn, timbl]) == 0:
        raise ValueError("Please select a classification experiment by setting the sklearn parameter, the timbl"
                         "parameter, or both to True.")

    # get the name of the corpus and create corpus specific file-names to store the derived co-occurrence matrix,
    # the statistics derived for the contexts, the statistics derived for the words, the weights computed for the files
    # specify the file extension of all these files, except the plot of the cosine similarities, which is gonna be saved
    # in .pdf
    ext = '.csv'
    corpus_name = os.path.splitext(os.path.basename(input_corpus))[0]
    similarities_plot = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'similarities']),
                                                          str(training_perc), '.pdf']))
    training_space = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'training']), str(training_perc), ext]))
    context_output = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'contextStats']), ext]))
    weights_file = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'featWeights']), str(training_perc), ext]))

    print()
    print(strftime("%Y-%m-%d %H:%M:%S") + ": Started creating the vector space from the training corpus.")

    # traverse the input corpus until the specified percentage that should be allocated for training:
    # derive a word-context co-occurrence matrix where rows are words and columns are contexts (all the unique rows and
    # unique contexts from the input corpus, mapped to numerical indices).
    # - training_co_occurrences is a 2dNumPy array storing word-context co-occurrences
    # - context_ids is a dictionary mapping contexts to numerical indices pointing to the column in
    #   training_co_occurrences corresponding to each context
    # - train_word_ids is a dictionary mapping contexts to numerical indices pointing to the row in
    #   training_co_occurrences corresponding to each word
    # - train_word_freqs is NumPy array with as many positions as there are rows in the co-occurrence matrix; each
    #   position stores the frequency count of the word in the corresponding row of the co-occurrence matrix
    train_co_occurrences, context_ids, train_word_ids, train_word_freqs = collect_contexts(input_corpus, training_perc,
                                                                                           pos_dict, bigrams=bigrams,
                                                                                           trigrams=trigrams)
    # print the co-occurrence matrix to file for later use
    print_vector_space(train_co_occurrences, train_word_ids, context_ids, training_space)

    print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished creating the vector space from the training corpus.")
    print()

    print()
    print(strftime("%Y-%m-%d %H:%M:%S") + ": Started computing context statistics from the training corpus.")

    # use TiMBL to compute feature weights for each context, based on the co-occurrence counts derived before
    feature_weights = compute_feature_weights(training_space, weights_file)

    # compute several distributional statistics for each context, and print them to a summary file (specified at the top
    # of this function)
    compute_context_statistics(train_co_occurrences, train_word_freqs, context_ids, feature_weights,
                               context_output, training_perc)

    print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished computing context statistics from the training corpus.")
    print()

    if not leave_one_out:
        # in case the train-test setting was selected (default) this block of the code is executed
        print()
        print(strftime("%Y-%m-%d %H:%M:%S") + ": Started creating the vector space from the test corpus.")

        # collect co-occurrence count in the portion of corpus allotted to the test set, only using the contexts
        # harvested during training: words that don't occur in any of these contexts are lost, and so are contexts that
        # didn't occur during training. The training and test matrix have to be perfectly aligned column-wise to
        # meaningfully compute similarities, hence the features collected at training are used also when processing the
        # test corpus.
        # - test_co_occurrences is analoguous to training_co_occurrences
        # - test_word_ids is analoguous to training_word_ids
        # - test_word_freqs is analoguous to train word_freqs
        test_space = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'test']), str(training_perc), ext]))
        test_co_occurrences, test_word_ids, test_word_freqs = create_test_space(input_corpus, test_perc, context_ids,
                                                                                pos_dict, bigrams=bigrams,
                                                                                trigrams=trigrams)
        print_vector_space(test_co_occurrences, test_word_ids, context_ids, test_space)
        target_word_frequencies = test_word_freqs
        target_co_occurrences = test_co_occurrences
        target_word_ids = test_word_ids

        print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished creating the vector space from the test corpus.")
        print()

    else:
        # in case the leave-one-out setting is chosen (by passing True to the 'leave_one_out' parameter, make all
        # structures pertaining to the test corpus None so that they're not considered in further functions. Also, use
        # word frequency counts, co-occurrence matrix, and word_ids derived from the training corpus
        test_word_ids, test_co_occurrences, test_space, context_ids = [None, None, None, None]
        target_word_frequencies = train_word_freqs
        target_co_occurrences = train_co_occurrences
        target_word_ids = train_word_ids

    print()
    print(strftime("%Y-%m-%d %H:%M:%S") + ": Started running the PoS tagging experiment and deriving word statistics.")

    accuracies = defaultdict(dict)
    if sklearn:
        # if sklearn is selected, use cosine as a distance metric and categorize target items (training words in the
        # leave-one-out setting and test words in the train-test setting), then compute statistics for the target words
        # using the appropriate co-occurrence matrix and corresponding row indices
        sklearn_accuracies, sim, word_ids = sklearn_experiment(train_co_occurrences, train_word_ids,
                                                               test_words=test_word_ids, test_space=test_co_occurrences,
                                                               contexts=context_ids, diag_value=diag_value,
                                                               nn=nn, plot=similarities_plot)
        for word in sklearn_accuracies:
            accuracies[word]['correct'] = sklearn_accuracies[word]['correct']
            accuracies[word]['sklearn'] = {'predicted': sklearn_accuracies[word]['predicted'],
                                           'accuracy': sklearn_accuracies[word]['accuracy']}

    if timbl:
        # if timbl is selected, use overlap as a distance metric and categorize target items (training words in the
        # leave-one-out setting and test words in the train-test setting), then compute statistics for the target words
        # using the appropriate co-occurrence matrix and corresponding row indices
        categorization_output = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'posTagged']),
                                                                  str(training_perc), ext]))
        timbl_accuracies = timbl_experiment(training_space, categorization_output, nn=nn, test=test_space)
        for word in timbl_accuracies:
            accuracies[word]['correct'] = timbl_accuracies[word]['correct']
            accuracies[word]['timbl'] = {'predicted': timbl_accuracies[word]['predicted'],
                                         'accuracy': timbl_accuracies[word]['accuracy']}

    # too many word_ids wrt to the words categorized in the experiment

    output = os.path.join(output_dir, ''.join(['_'.join([corpus_name, 'wordStats']), ext]))
    compute_words_statistics(target_co_occurrences, target_word_ids, target_word_frequencies, accuracies,
                             feature_weights, output, training_perc)

    print(strftime("%Y-%m-%d %H:%M:%S") + ": Finished running the PoS tagging experiment and deriving word statistics.")
    print()


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description="Select distributional contexts and perform "
                                                 "a PoS tagging experiment using kNN clustering.")

    parser.add_argument("-c", "--corpus", required=True, dest="corpus",
                        help="Specify the corpus file to be used as input (encoded as a .json file).")
    parser.add_argument("-o", "--output_folder", required=True, dest="output_folder",
                        help="Specify the folder where output files will be stored.")
    parser.add_argument("-p", "--pos_file", required=True, dest="pos_file",
                        help="Specify the path to the file containing the PoS mapping from CHILDES tags to custom"
                             "coarser tags.")
    parser.add_argument("-l", "--leave_one_out", action="store_false", dest="leave_one_out",
                        help="Specify whether to use a leave-one-out setting where words from the training set are"
                             "categorized searching the same set of words (except for the target) for the nearest "
                             "neighbours. Default is true: selecting the option -l causes the program NOT to run"
                             "a leave-one-out experiment.")
    parser.add_argument("-n", "--nearest_neighbours", dest="nn", default=1,
                        help="Specify the number of nearest neighbours to be considered for categorization.")
    parser.add_argument("-d", "--diagonal", dest="diag_value", default=None,
                        help="Specify the value to which the main diagonal of the cosine matrix is set."
                             "Default is None meaning that nothing happens; by setting the value to 0, categorization"
                             "is forced to discard same word types when finding nearest neighbours, ensuring"
                             "generalization.")
    parser.add_argument("-m", "--minimum_training", dest="min_train", default=40,
                        help="Specify the minimum percentage of utterances from each input corpus to be used for "
                             "training (default is 40).")
    parser.add_argument("-s", "--steps", dest="steps", default=7,
                        help="Specify the number of training steps (default is 7, meaning that the model is "
                             "trained first using 40% of the utterances, then 45%, then 50%, and so on, up to 70%,"
                             "resulting in 7 steps).")
    parser.add_argument("-T", "--test_perc", dest="test_perc", default=30,
                        help="Specify the percentage of utterances from each input corpus to be used for testing "
                             "(default is 30). The sum of the value passed to -T and to -M need to add to 100.")
    parser.add_argument("-b", "--bigrams", action="store_true", dest="bigrams",
                        help="Specify whether to consider bigrams as relevant distributional contexts.")
    parser.add_argument("-t", "--trigrams", action="store_true", dest="trigrams",
                        help="Specify whether to consider trigrams as relevant distributional contexts.")
    parser.add_argument("-K", "--sklearn", action="store_true", dest="sklearn",
                        help="Specify whether to use sklearn cosine computation to perform the PoS tagging experiment.")
    parser.add_argument("-B", "--timbl", action="store_true", dest="timbl",
                        help="Specify whether to use TiMBL to perform the PoS tagging experiment.")

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    max_train = 100 if args.leave_one_out else 100 - int(args.test_perc)
    min_train = int(args.min_train)
    steps = int(args.steps)
    test_perc = int(args.test_perc)
    diag_value = args.diag_value if args.diag_value is None else int(args.diag_value)
    pos_dict = read_category_mapping(args.pos_file)
    training_steps = np.linspace(min_train, max_train, num=steps)
    for train_perc in training_steps:
        context_analysis(args.corpus, pos_dict, args.output_folder, training_perc=int(train_perc), test_perc=test_perc,
                         bigrams=args.bigrams, trigrams=args.trigrams, diag_value=diag_value,
                         nn=int(args.nn), sklearn=args.sklearn, timbl=args.timbl, leave_one_out=args.leave_one_out)


########################################################################################################################


if __name__ == '__main__':

    main()
