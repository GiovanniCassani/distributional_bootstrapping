__author__ = 'GCassani'

 """Compute correlations between Information Gain and Salience, and between predictability and average 
    conditional probability of contexts given words."""

import os
import re
import argparse
import pandas as pd
from context_utils.readers import read_category_mapping
from context_utils.correlations.ig2salience import ig2salience
from context_utils.correlations.pred2condprob import pred2condprob
from context_utils.cumulative_learning.model_id import make_model_id
from context_utils.cumulative_learning.corpus_section import make_corpus_section


def compute_correlations(corpus_folder, output_folder, pos_map_file, threshold=0, boundaries=True,
                         bigrams=True, trigrams=True, pred=True, div=True, freq=True, averages=True):

    pos_map_file = os.path.abspath(pos_map_file)
    pos_mapping = read_category_mapping(pos_map_file)

    # create the output folder if it doesn't exist and get the absolute paths of both folders
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    t = ''.join(['k', str(threshold)])
    avgs = 'avgs' if averages else 'no-avgs'

    output_folder = os.path.abspath(os.path.join(output_folder, avgs, t))
    corpus_folder = os.path.abspath(corpus_folder)

    ig2salience_df = pd.DataFrame(columns=['model', 'boundaries', 'corpus', 'age', 'time',
                                           'threshold', 'avgs', 'corr_ig2salience'])
    pred2prob_df = pd.DataFrame(columns=['model', 'boundaries', 'corpus', 'age', 'time',
                                         'threshold', 'avgs', 'corr_pred2prob'])

    corpora = sorted(list(os.walk(corpus_folder))[0][1])
    for corpus in corpora:

        # for each corpus, define the path to the folder that contains the data (corpus_input_dir) and to the folder
        # that will contain results and working files (corpus_output_folder)
        corpus_input_dir = os.path.join(corpus_folder, corpus)
        corpus_output_folder = os.path.join(output_folder, corpus)
        if not os.path.exists(corpus_output_folder):
            os.makedirs(corpus_output_folder)

        # define a regular expression to identify file names that contain numbers and only take such files from the
        # folder of data from a corpus
        num = re.compile(r'\d+')
        age_sections = [x for x in sorted(list(os.walk(corpus_input_dir))[0][2]) if num.findall(x)]
        months = len(age_sections)

        for i in range(months):

            # get the current age
            age = num.findall(age_sections[i])[0]

            # make a sub-folder for the current age in the output folder of the current corpus
            section_output_folder = os.path.join(corpus_output_folder, os.path.splitext(age_sections[i])[0])
            if not os.path.exists(section_output_folder):
                os.makedirs(section_output_folder)
            if os.getcwd() != corpus_input_dir:
                os.chdir(corpus_input_dir)

            # collapse all corpus sub-sections up to the current age to make the training section, and compute the
            # number of sentences, tokens, and types in it
            training_corpus = os.path.join(section_output_folder, 'training_set.txt')
            make_corpus_section('training_set.txt', age_sections, i,
                                target_dir=section_output_folder, pos_dict=pos_mapping)
            print("Created training corpus for age %s in corpus %s." % (age, corpus))
            os.chdir(corpus_folder)

            model = make_model_id(pred=pred, div=div, freq=freq, bigrams=bigrams, trigrams=trigrams)
            model_output_folder = os.path.join(section_output_folder, 'boundaries', model) if boundaries else \
                os.path.join(section_output_folder, 'no_boundaries', model)

            corr_ig2salience = ig2salience(training_corpus, model_output_folder, pos_dict=pos_mapping, k=threshold,
                                           bigrams=bigrams, trigrams=trigrams, boundaries=boundaries,
                                           pred=pred, div=div, freq=freq, averages=averages)
            ig2salience_df = ig2salience_df.append({'model': model,
                                                    'boundaries': boundaries,
                                                    'corpus': corpus,
                                                    'age': age,
                                                    'time': i,
                                                    'threshold': threshold,
                                                    'avgs': averages,
                                                    'corr_ig2salience': corr_ig2salience}, ignore_index=True)

            corr_pred2prob = pred2condprob(training_corpus, model_output_folder, pos_dict=pos_mapping, k=threshold,
                                           bigrams=bigrams, trigrams=trigrams, boundaries=boundaries,
                                           pred=pred, div=div, freq=freq, averages=averages)
            pred2prob_df = pred2prob_df.append({'model': model,
                                                'boundaries': boundaries,
                                                'corpus': corpus,
                                                'age': age,
                                                'time': i,
                                                'threshold': threshold,
                                                'avgs': averages,
                                                'corr_pred2prob': corr_pred2prob}, ignore_index=True)

    ig2salience_file = os.path.join(output_folder, 'ig2salience.csv')
    if not os.path.exists(ig2salience_file):
            ig2salience_df.to_csv(ig2salience_file, sep='\t', index=False)
    else:
        with open(ig2salience_file, 'a') as f:
            ig2salience_df.to_csv(f, sep='\t', index=False, header=False)

    pred2prob_file = os.path.join(output_folder, 'pred2prob.csv')
    if not os.path.exists(pred2prob_file):
        pred2prob_df.to_csv(pred2prob_file, sep='\t', index=False)
    else:
        with open(pred2prob_file, 'a') as f:
            pred2prob_df.to_csv(f, sep='\t', index=False, header=False)

    return ig2salience_df, pred2prob_df


########################################################################################################################


def main():

    parser = argparse.ArgumentParser(description="Check correlations between IG and salience, and between"
                                                 "predictability and average conditional probability")

    parser.add_argument("-c", "--corpus_folder", required=True, dest="corpus_folder",
                        help="Specify the folder where longitudinally spliced corpora are stored (as folders).")
    parser.add_argument("-o", "--output_folder", required=True, dest="output_folder",
                        help="Specify the folder where summary files and experiments' output files are stored.")
    parser.add_argument("-m", "--mapping", required=True, dest="pos_mapping",
                        help="Specify the path to the file containing the mapping between CHILDES and custom PoS tags.")
    parser.add_argument("-k", "--salience_threshold", dest="k", default=1,
                        help="Set the threshold to decide which contexts are salient.")
    parser.add_argument("-u", "--utterance", action="store_true", dest="boundaries",
                        help="Specify whether to consider utterance boundaries.")
    parser.add_argument("-b", "--bigrams", action="store_true", dest="bigrams",
                        help="Specify whether to consider bigrams.")
    parser.add_argument("-t", "--trigrams", action="store_true", dest="trigrams",
                        help="Specify whether to consider trigrams.")
    parser.add_argument("-p", "--predictability", action="store_true", dest="predictability",
                        help="Specify whether to consider predictability in deciding about contexts' relevance.")
    parser.add_argument("-d", "--diversity", action="store_true", dest="diversity",
                        help="Specify whether to consider lexical diversity in deciding about contexts' relevance.")
    parser.add_argument("-f", "--frequency", action="store_true", dest="frequency",
                        help="Specify whether to consider frequency in deciding about contexts' relevance.")
    parser.add_argument("-a", "--averages", action="store_true", dest="averages",
                        help="Specify whether to compare frequency, diversity, and predictability scores to running"
                             "averages, or not.")

    args = parser.parse_args()

    compute_correlations(args.corpus_folder,
                         args.output_folder,
                         args.pos_mapping,
                         threshold=int(args.k),
                         boundaries=args.boundaries,
                         bigrams=args.bigrams,
                         trigrams=args.trigrams,
                         pred=args.predictability,
                         div=args.diversity,
                         freq=args.frequency,
                         averages=args.averages)


########################################################################################################################


if __name__ == '__main__':

    main()
