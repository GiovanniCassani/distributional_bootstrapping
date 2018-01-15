__author__ = 'GCassani'

"""Compute correlation between Information Gain and salience"""


import os
import numpy as np
import pandas as pd
from context_utils.contexts.salient_contexts import get_useful_contexts
from context_utils.regression_analysis.contexts.feature_weights import compute_feature_weights
from context_utils.vector_spaces.maker import create_vector_space
from context_utils.vector_spaces.printer import print_vector_space


def ig2salience(input_corpus, output_folder, pos_dict=None, k=0, boundaries=True, bigrams=True, trigrams=True,
                pred=True, div=True, freq=True, averages=True):

    """
    :param input_corpus:    the path to a .txt file containing CHILDES transcripts, with one utterance per line and
                            words divided by white spaces. The first element of each utterance is the capitalized label
                            of the speaker, as found in CHILDES. The second element is a dummy word marking the
                            beginning of the utterance, #start; the last element is a dummy word marking the end of the
                            utterance, #end. Each word is paired to its Part-of-Speech tag, the two separated by a
                            tilde, word~PoS.
    :param output_folder:   a string indicating the folder where simulation results will be stored
    :param pos_dict:        a dictionary mapping CHILDES PoS tags to custom tags; any CHILDES tag that doesn't appear as
                            key in the dictionary is assumed to be irrelevant and words labeled with those tags are
                            discarded; the default to None means that every input word is considered
    :param k:               the threshold to determine which contexts are salient: every context whose score is higher
                            than t is considered to be salient. The default is 1.
    :param boundaries:      a boolean indicating whether utterance boundaries are to be considered or not as context
    :param bigrams:         a boolean indicating whether bigrams are to be collected
    :param trigrams:        a boolean indicating whether trigrams are to be collected
    :param pred:            a boolean indicating whether average predictability of contexts given words are
                            a relevant piece of information in deciding about how important a context it
    :param div:             a boolean indicating whether lexical diversity of contexts, i.e. the number of different
                            words that occur in a context, is a relevant piece of information in deciding about how
                            important a context it
    :param freq:            a boolean indicating whether contexts' frequency count is a relevant piece of information in
                            deciding about how important a context it
    :param averages:        a boolean specifying whether frequency, diversity, and predictability values for each
                            context have to be compared to running averages
    :return correlation:    the Pearson correlation between the IG values and the salience values for each context
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    training_file = '/'.join([output_folder, 'training.csv'])
    weights_file = '/'.join([output_folder, 'feature_weights.csv'])
    output_file = '/'.join([output_folder, 'ig2salience.csv'])

    salient_contexts = get_useful_contexts(input_corpus, pos_dict=pos_dict, k=k, boundaries=boundaries,
                                           bigrams=bigrams, trigrams=trigrams, pred=pred, div=-div, freq=freq,
                                           averages=averages)
    co_occurrences, useless, unused, word_ids, context_ids = create_vector_space(input_corpus, salient_contexts,
                                                                                 pos_dict=pos_dict,
                                                                                 boundaries=boundaries,
                                                                                 bigrams=bigrams, trigrams=trigrams)
    print_vector_space(co_occurrences, word_ids, context_ids, training_file)
    feature_weights = compute_feature_weights(training_file, weights_file)

    # make a Pandas data frame and populate it: each row consists of the context string, its IG value and its salience
    # as computed inside the get_useful_contexts function
    df = pd.DataFrame(index=np.arange(0,len(feature_weights)),
                      columns=['context','ig','salience'])
    for context, idx in context_ids.items():
        ig = feature_weights[idx]['ig']
        salience = salient_contexts[context]
        df.loc[idx] = [context, ig, salience]
    df.to_csv(output_file, sep='\t', index=False)

    df['ig'] = np.float64(df['ig'])
    df['salience'] = np.float64(df['salience'])

    correlation = df['ig'].corr(df['salience'])

    return correlation
