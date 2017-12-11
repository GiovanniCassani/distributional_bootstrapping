__author__ = 'GCassani'


import os
import re
import argparse
import operator
import context_utils as ctxt
from collections import defaultdict


########################################################################################################################


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


########################################################################################################################


def make_corpus_section(output_file, ages, i, target_dir='', pos_dict=None, training=True):

    """
    :param output_file:     the path to the file that will be created
    :param ages:            a list of file-names
    :param i:               an index (between 0 and the length of the input list
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
    sentences = ctxt.count_cds_lines(destination)
    tokens = ctxt.count_cds_tokens(destination, pos_dict=pos_dict)
    types = ctxt.count_cds_unique_words(destination, pos_dict=pos_dict)

    return sentences, tokens, types


########################################################################################################################


def print_contexts(contexts, path):

    """
    :param contexts:    a dictionary mapping strings to numbers
    :param path:        the path to the file where contexts are gonna be printed
    """

    with open(path, 'w+') as f:
        for k in sorted(contexts.items(), key=operator.itemgetter(1), reverse=True):
            f.write('\t'.join([k[0], str(k[1])]))
            f.write('\n')


########################################################################################################################


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


########################################################################################################################


def cumulative_learning(corpus_folder, output_folder, pos_map_file,
                        leave_one_out=True, nn=1, timbl=False, sklearn=True, threshold=1.0, diag_value=None,
                        training_months=float('inf'), boundaries=True, bigrams=True, trigrams=True,
                        freq_frames=True, flex_frames=False, cond_prob=True, div=True, freq=True, averages=False,
                        target_tr_path='', target_te_path='', ignore_tr_path='', ignore_te_path=''):

    """
    :param corpus_folder:       a string indicating in which folder the corpora for each child are to be found
    :param output_folder:       a string indicating the folder where simulation results will be stored
    :param pos_map_file:        a string indicating the path to the file containing the mapping from CHILDES PoS tags to
                                coarser, custom tags is to be found. The file consists of two, space separated columns:
                                the first contains the CHILDES tags, the second the corresponding custom tags. If a
                                CHILDES tag is not present, all words tagged with it are not considered
    :param leave_one_out:       a boolean to specify whether to use a leave-one-out setting where words from the
                                training set are categorized searching the same set of words (except for the target) for
                                the nearest neighbours. This is the default setting.
    :param nn:                  the number of nearest neighbours to be considered when classifying a word from the test
                                set
    :param timbl:               a boolean specifying whether TiMBL must be used for categorization
    :param sklearn:             a boolean specifying whether sklearn, cosine-based kNN classifier must be used
    :param diag_value:          the value to which all the cells on the main diagonal of the matrix of cosine
                                similarities between test and training vectors are set (default is 0, meaning that cells
                                on the main diagonal don't impact the nearest neighbour computation). This option makes
                                it possible to force the model to categorize a test word while ignoring the vector from
                                the training space that correspond to the same word type, thus enforcing generalization.
    :param threshold:           the threshold to determine which contexts are salient in the experimental model: every
                                context whose score is higher than this calue is considered to be salient. The default
                                is 1 (see the paper to know why).
    :param diag_value:          the value to which the diagonal cells of the matrix of similarities in the sklearn
                                categorization experiment are set; default is None, meaning that the matrix of
                                similarities if left untouched
    :param training_months:     the number of months from the input corpora to be used for training (the default is 8,
                                which offers a good balance between training and test section for the children in the
                                Manchester corpus)
    :param boundaries:          a boolean indicating whether the model should consider utterance boundaries as relevant
                                distributional contexts
    :param bigrams:             a boolean indicating whether bigram contexts need to be considered
    :param trigrams:            a boolean indicating whether trigram contexts need to be considered
    :param freq_frames:         a boolean indicating whether the frequent frame model should be run
    :param flex_frames:         a boolean indicating whether the flexible frame model should be run
    :param cond_prob:           a boolean indicating whether the experimental model should decide on contexts' relevance
                                considering the average conditional probability of contexts given words
    :param div:                 a boolean indicating whether the experimental model should decide on contexts' relevance
                                considering the lexical diversity of a context, i.e. the number of different words a
                                context co-occurs with
    :param freq:                a boolean indicating whether the experimental model should decide on contexts' relevance
                                considering the frequency count of a context
    :param averages:            a boolean specifying whether frequency, diversity, and predictability scores have to be
                                compared to running averages or not. If this argument is set to True, the three pieces
                                of information are considered as relative: more/less frequent than the average context,
                                more/less diverse than the average context, and so on. Thus, ratios are multiplied, with
                                numbers higher than 1 indicating contexts that are more frequent/diverse/predictable
                                than average, and viceversa. If this argument is set to False, scores are combined as
                                such
    :param target_tr_path:      a string indicating the path to the file containing the words (one word per line,
                                formatted like this: 'word~PoS' with the original PoS tags from CHILDES) to be used as
                                targets when collecting co-occurrence counts on the training set. The default is the
                                empty string meaning that all words from the training set are considered as targets. If
                                a file is passed, only the words in it are considered.
    :param target_te_path:      same as for target_train_path, but to be applied to the test set. Training and test
                                targets are separated to allow to dissociate the two.
    :param ignore_tr_path:      similar to target_train_path and target_test_path, but instead of providing the list of
                                target words, this provides the list of words to be ignored. The default is the empty
                                string, meaning that all words from the training set are considered. If a file is passed
                                the words in it are not considered.
    :param ignore_te_path:      same as for ignore_test_path, but applied to the test set.
    :return categorization:     a dictionary that maps a binding of the following information to a boolean indcating
                                whether the word was categorized correctly (1) or not (0)
                                - the model that generated the categorization
                                - whether the model was sensitive to utterance boundaries (1) or not (0)
                                - the age at which the word was categorized
                                - the time steps, derived from the age but on an ordinal rather than ratio scale
                                - the corpus where the word was found and categorized
                                - the word string, together with the corresponding CHILDES PoS tag
                                - the correct PoS tag in the corpus (from the custom tagset)
                                - the predicted PoS tag (from the custom target)
    :return summary:            a dictionary mapping a binding of the following information to an accuracy value
                                indicating the effectiveness of PoS tagging:
                                - the model that generated the categorization
                                - whether the model was sensitive to utterance boundaries (1) or not (0)
                                - the age at which the word was categorized
                                - the time steps, derived from the age but on an ordinal rather than ratio scale
                                - the corpus where the word was found and categorized
                                - the number of utterances in the training set
                                - the number of tokens in the training set
                                - the number of word types in the training set
                                - the number of utterances in the test set
                                - the number of tokens in the test set
                                - the number of word types in the test set
                                - the number of contexts selected as important by the model
                                - the proportion of salient contexts that were not found in the test set, or that could
                                    not be used to categorize any word because only occurred with a single word type
                                - the proportion of word types from the test set that never occurred in any of the
                                    salient contexts
    """

    if any([target_tr_path, target_te_path, ignore_tr_path, ignore_te_path]) and leave_one_out:
        raise ValueError('Cannot combine the leave-one-out experimental setting with sets of target words or'
                         'words to be ignored in training and test, as there are no train and test corpora with'
                         'the leave-one-out setting.')

    if sum([freq_frames, flex_frames, any([cond_prob, freq, div])]) == 0:
        raise ValueError('No model specified! Please set either freq_frames, flex_frames, or at least one among'
                         'freq, div, and cond_prob parameters to True for the model to run.')

    if sum([freq_frames, flex_frames, any([cond_prob, freq, div])]) > 1:
        raise ValueError('Too many active models! This function can select contexts according to three models:'
                         'frequent frames, flexible frames, and the CGDG model relying on frequency, diversity, and'
                         'conditional probability. It looks like you activated at least two of these three models'
                         'by setting the freq_frames, flex_frames, and one of the freq, div, and cond_prob parameters'
                         'to True.')

    target_tr_path = os.path.abspath(target_tr_path) if target_tr_path else ''
    target_te_path = os.path.abspath(target_te_path) if target_te_path else ''
    ignore_tr_path = os.path.abspath(ignore_tr_path) if ignore_tr_path else ''
    ignore_te_path = os.path.abspath(ignore_te_path) if ignore_te_path else ''

    pos_map_file = os.path.abspath(pos_map_file)
    pos_mapping = ctxt.read_category_mapping(pos_map_file)

    # create the output folder if it doesn't exist and get the absolute paths of both folders
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder = os.path.abspath(output_folder)
    corpus_folder = os.path.abspath(corpus_folder)

    # define the paths to the two output files that will contain summary information for the subsequent
    # statistical analysis
    accuracies, summary = [defaultdict(dict), defaultdict(dict)]
    summary_file = '/'.join([output_folder, 'summary.txt'])
    categorization_file = '/'.join([output_folder, 'categorization.txt'])

    # get the sub-folders in the input folder
    corpora = sorted(list(os.walk(corpus_folder))[0][1])
    for corpus in corpora:

        print(corpus)

        # for each corpus, define the path to the folder that contains the data (corpus_input_dir) and to the folder
        # that will contain results and working files (corpus_output_folder)
        corpus_input_dir = '/'.join([corpus_folder, corpus])
        corpus_output_folder = '/'.join([output_folder, corpus])
        if not os.path.exists(corpus_output_folder):
            os.makedirs(corpus_output_folder)

        # define a regular expression to identify file names that contain numbers and only take such files from the
        # folder of data from a corpus
        num = re.compile(r'\d+')
        age_sections = [x for x in sorted(list(os.walk(corpus_input_dir))[0][2]) if num.findall(x)]
        months = len(age_sections) if training_months == float('inf') else training_months
        os.chdir(corpus_input_dir)

        if not leave_one_out:
            # make the test section, that will always be the same for every corpus, and compute the number of sentences,
            # tokens, and types in it
            test_corpus = '/'.join([corpus_output_folder, 'test_set.txt'])
            test_sentences, test_tokens, test_types = make_corpus_section('test_set.txt', age_sections, training_months,
                                                                          target_dir=corpus_output_folder,
                                                                          pos_dict=pos_mapping, training=False)
        else:
            test_corpus = None
            test_sentences, test_tokens, test_types = [float('nan'), float('nan'), float('nan')]

        # loop through the sections of each corpus, in chronological order (hence the need to look at numbers in the
        # file names)
        for i in range(months):

            # get the current age
            age = num.findall(age_sections[i])[0]

            # make a sub-folder for the current age in the output folder of the current corpus
            section_output_folder = '/'.join([corpus_output_folder, os.path.splitext(age_sections[i])[0]])
            if not os.path.exists(section_output_folder):
                os.makedirs(section_output_folder)
            if os.getcwd() != corpus_input_dir:
                os.chdir(corpus_input_dir)

            # collapse all corpus sub-sections up to the current age to make the training section, and compute the
            # number of sentences, tokens, and types in it
            training_corpus = '/'.join([section_output_folder, 'training_set.txt'])
            train_sentences, train_tokens, train_types = make_corpus_section('training_set.txt', age_sections, i,
                                                                             target_dir=section_output_folder,
                                                                             pos_dict=pos_mapping)
            print("Created training corpus for age %s in corpus %s." % (age, corpus))
            os.chdir(corpus_folder)

            if freq_frames:
                model = 'freqFrames'
                selected = 45

                # get the frequent frames for the training corpus
                frames = ctxt.harvest_frames(training_corpus, pos_dict=pos_mapping, boundaries=boundaries,
                                             freq_frames=True, flex_frames=False)
                salient_contexts = ctxt.get_salient_frames(frames, selected)
                print('Estimated frequent frames for corpus %s up to age %s' % (corpus, age))

            elif flex_frames:
                model = 'flexFrames'
                selected = 90

                frames = ctxt.harvest_frames(training_corpus, pos_dict=pos_mapping, boundaries=boundaries,
                                             freq_frames=False, flex_frames=True)
                salient_contexts = ctxt.get_salient_frames(frames, selected)
                print('Estimated flexible frames for corpus %s up to age %s' % (corpus, age))

            elif any([cond_prob, div, freq]):
                model = ctxt.model_id(cond_prob=cond_prob, div=div, freq=freq, bigrams=bigrams, trigrams=trigrams)
                salient_contexts = ctxt.learning_contexts(training_corpus, pos_dict=pos_mapping, k=threshold,
                                                          boundaries=boundaries, bigrams=bigrams, trigrams=trigrams,
                                                          cond_prob=cond_prob, div=div, freq=freq, averages=averages)
                selected = len(salient_contexts) if len(salient_contexts) else float('nan')
                print('Estimated salient contexts for corpus %s up to age %s' % (corpus, age))

            else:
                raise ValueError('Unrecognized model!')

            # make a sub-folder for the model in the age sub-folder of the corpus sub-folder in the output folder
            # for any choice wrt utterance boundaries. Essentially, in each corpus folder (output) there
            # will be a sub-folder for each age, and in each age sub-folder there will be a folder for each model;
            # in each model sub-folder, there will be a folder for models with and without utterance boundaries, and
            # in them a sub-folder for every model
            model_output_folder = '/'.join([section_output_folder, 'boundaries', model]) if boundaries else \
                '/'.join([section_output_folder, 'no_boundaries', model])
            if not os.path.exists(model_output_folder):
                os.makedirs(model_output_folder)

            training_file = '/'.join([model_output_folder, 'training_file.csv'])
            print_contexts(salient_contexts, '/'.join([model_output_folder, 'salientContexts.txt']))
            b = str(1) if boundaries else str(0)

            # build training and test vector spaces using the contexts deemed salient according to the model being run
            # the x variable stands for the contexts ids, which are however useless in the leave-one-out design;
            # if the training-test setting is chosen, context_ids will be derived when constructing the test space
            training, useless, unused, tr_words, context_ids = ctxt.build_experiment_vecspace(training_corpus,
                                                                                              salient_contexts,
                                                                                              pos_dict=pos_mapping,
                                                                                              boundaries=boundaries,
                                                                                              bigrams=True,
                                                                                              trigrams=True,
                                                                                              targets=target_tr_path,
                                                                                              to_ignore=ignore_tr_path)
            if training.any():
                ctxt.create_vector_space(training, tr_words, context_ids, training_file)
                print('Created training space for corpus %s up to age %s, in folder %s' %
                      (corpus, age, model_output_folder))

                if not leave_one_out:
                    test_file = '/'.join([model_output_folder, 'test_file.csv'])
                    # the x variable would be the same as tr_contexts since the contexts don't change: to mark the
                    # uselessness of the variable, I set it to x
                    test, useless, unused, te_words, context_ids = ctxt.build_experiment_vecspace(test_corpus,
                                                                                                  salient_contexts,
                                                                                                  pos_dict=pos_mapping,
                                                                                                  boundaries=boundaries,
                                                                                                  bigrams=True,
                                                                                                  trigrams=True,
                                                                                                  targets=target_te_path,
                                                                                                  to_ignore=ignore_te_path)
                    types = test_types
                    if test.any():
                        ctxt.create_vector_space(test, te_words, context_ids, test_file)
                        print('Created test space using contexts estimate up to age %s for corpus %s, with model %s' %
                              (age, corpus, model))

                else:
                    test, te_words, test_file, context_ids = [None, None, None, None]
                    types = train_types

                coverage = len(unused) / types
                proportion_useless = len(useless) / selected
                s_key = '\t'.join([model, b, age, str(i), corpus,
                                   str(train_sentences), str(train_tokens), str(train_types),
                                   str(test_sentences), str(test_tokens), str(test_types),
                                   str(selected), str(proportion_useless), str(coverage)])

                if timbl:
                    # run the TiMBL experiment and get accuracy scores; finally update the appropriate structures
                    output_file = '/'.join([model_output_folder, 'output_space.txt'])
                    timbl_accuracies = ctxt.timbl_experiment(training_file, output_file, test=test_file, nn=nn)
                    accuracies = update_categorization_dict(timbl_accuracies, unused, accuracies, 'timbl', model, b,
                                                            age, i, corpus)
                    timbl_stats = ctxt.category_f1(timbl_accuracies)
                    summary[s_key]['timbl'] = str(timbl_stats['all']['f1'])

                if sklearn:
                    # run the sklearn experiment (co-occurrence matrices are printed inside the run_sklearn_experiment
                    # function) and get accuracy scores; finally update the appropriate structures
                    plot_path = '/'.join([model_output_folder, 'similarities.pdf'])
                    sklearn_accuracies, sim, word_ids = ctxt.sklearn_cosine_similarity(training, tr_words,
                                                                                       test_space=test,
                                                                                       test_words=te_words,
                                                                                       contexts=context_ids,
                                                                                       nn=nn, diag_value=diag_value,
                                                                                       plot=plot_path)
                    accuracies = update_categorization_dict(sklearn_accuracies, unused, accuracies, 'sklearn', model,
                                                            b, age, i, corpus)
                    sklearn_stats = ctxt.category_f1(sklearn_accuracies)
                    summary[s_key]['sklearn'] = str(sklearn_stats['all']['f1'])

                print('Run PoS tagging experiment and evaluated it for corpus %s up to age %s' % (corpus, age))

    if not os.path.exists(summary_file):
        make_summary_file(summary_file)
    with open(summary_file, 'a+') as s_f:
        for item in summary:
            timbl_acc = get_accuracy(summary, item, 'timbl')
            sklearn_acc = get_accuracy(summary, item, 'sklearn')
            s_f.write('\t'.join([item, timbl_acc, sklearn_acc]))
            s_f.write('\n')

    if not os.path.exists(categorization_file):
        make_categorization_file(categorization_file)
    with open(categorization_file, 'a+') as c_f:
        for item in accuracies:
            correct = accuracies[item]['correct']
            timbl_predicted, timbl_accuracy = get_predictions(accuracies, item, 'timbl')
            sklearn_predicted, sklearn_accuracy = get_predictions(accuracies, item, 'sklearn')
            c_f.write('\t'.join([item, correct,
                                 timbl_predicted, str(timbl_accuracy),
                                 sklearn_predicted, str(sklearn_accuracy)]))
            c_f.write('\n')

    return accuracies, summary


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description="Select distributional contexts and perform "
                                                 "a PoS tagging experiment using TiMBL.")

    parser.add_argument("-c", "--corpus_folder", required=True, dest="corpus_folder",
                        help="Specify the folder where longitudinally spliced corpora are stored (as folders).")
    parser.add_argument("-o", "--output_folder", required=True, dest="output_folder",
                        help="Specify the folder where summary files and experiments' output files are stored.")
    parser.add_argument("-m", "--mapping", required=True, dest="pos_mapping",
                        help="Specify the path to the file containing the mapping between CHILDES and custom PoS tags.")
    parser.add_argument("-l", "--leave_one_out", action="store_false", dest="leave_one_out",
                        help="Specify whether to use a leave-one-out setting where words from the training set are"
                             "categorized searching the same set of words (except for the target) for the nearest "
                             "neighbours. Default is true: selecting the option -l causes the program NOT to run"
                             "a leave-one-out experiment.")
    parser.add_argument("-n", "--nearest_neighbors", dest="nn", default=1,
                        help="Set the number of nearest neighbors to consider when categorizing a test word.")
    parser.add_argument("-K", "--sklearn", action="store_true", dest="sklearn",
                        help="Specify whether to use kNN cosine-based classifier implemented in sklearn.")
    parser.add_argument("-B", "--timbl", action="store_true", dest="timbl",
                        help="Specify whether to use TiMBL numeric overlap classification.")
    parser.add_argument("-k", "--salience_threshold", dest="k", default=1,
                        help="Set the threshold to decide which contexts are salient.")
    parser.add_argument("-D", "--diag_value", dest="diag_value", default=None,
                        help="Set the number of the cells in the main diagonal of the cosine similarity matrix of"
                             "training and test vectors (default is None, meaning that the diagonal is left untouched.")
    parser.add_argument("-M", "--months", dest="training_months", default=float('inf'),
                        help="Specify how many corpora sub-sections to consider for incremental training. The default "
                             "is set to infinite, meaning that all available months are used for training in a"
                             "leave-one-out categorization experiment.")
    parser.add_argument("-u", "--utterance", action="store_true", dest="boundaries",
                        help="Specify whether to consider utterance boundaries.")
    parser.add_argument("-b", "--bigrams", action="store_true", dest="bigrams",
                        help="Specify whether to consider bigrams.")
    parser.add_argument("-r", "--trigrams", action="store_true", dest="trigrams",
                        help="Specify whether to consider trigrams.")
    parser.add_argument("-F", "--freq_frames", action="store_true", dest="freq_frames",
                        help="Specify whether to consider trigrams.")
    parser.add_argument("-f", "--flex_frames", action="store_true", dest="flex_frames",
                        help="Specify whether to consider bigrams.")
    parser.add_argument("-p", "--probability", action="store_true", dest="cond_probability",
                        help="Specify whether to consider cond. probability in deciding about contexts' relevance.")
    parser.add_argument("-d", "--diversity", action="store_true", dest="diversity",
                        help="Specify whether to consider lexical diversity in deciding about contexts' relevance.")
    parser.add_argument("-q", "--frequency", action="store_true", dest="frequency",
                        help="Specify whether to consider frequency in deciding about contexts' relevance.")
    parser.add_argument("-a", "--averages", action="store_true", dest="averages",
                        help="Specify whether to compare frequency, diversity, and predictability scores to running"
                             "averages, or not.")
    parser.add_argument("-t", "--train_targets", dest="train_targets", default="",
                        help="Specify the path to the file containing the words to be considered"
                             "as targets during training (default: no file).")
    parser.add_argument("-T", "--test_targets", dest="test_targets", default="",
                        help="Specify the path to the file containing the words to be categorized at test "
                             "(default: no file).")
    parser.add_argument("-i", "--train_ignore", dest="train_ignore", default="",
                        help="Specify the path to the file containing the words to be ignored in the training "
                             "corpus (default: no file).")
    parser.add_argument("-I", "--test_ignore", dest="test_ignore", default="",
                        help="Specify the path to the file containing the words to be ignored in the test "
                             "corpus (default: no file).")

    args = parser.parse_args()

    training_months = int(args.training_months) if args.training_months != float('inf') else args.training_months

    diag_value = None if args.diag_value is None else int(args.diag_value)

    cumulative_learning(args.corpus_folder,
                        args.output_folder,
                        args.pos_mapping,
                        leave_one_out=args.leave_one_out,
                        nn=int(args.nn),
                        timbl=args.timbl,
                        sklearn=args.sklearn,
                        threshold=int(args.k),
                        diag_value=diag_value,
                        training_months=training_months,
                        boundaries=args.boundaries,
                        bigrams=args.bigrams,
                        trigrams=args.trigrams,
                        freq_frames=args.freq_frames,
                        flex_frames=args.flex_frames,
                        cond_prob=args.cond_probability,
                        div=args.diversity,
                        freq=args.frequency,
                        averages=args.averages,
                        target_tr_path=args.train_targets,
                        target_te_path=args.test_targets,
                        ignore_tr_path=args.train_ignore,
                        ignore_te_path=args.test_ignore)


########################################################################################################################


if __name__ == '__main__':

    main()
