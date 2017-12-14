from collections import defaultdict


def get_precision_and_recall(hits, category):

    """
    :param hits:        a dictionary of dictionaries mapping a word to three fields:
                        hits[word]['predicted'] gives the PoS tag predicted by the model for the word
                        hits[word]['correct'] gives the correct PoS tag for the word
                        hits[word]['accuracy'] tells whether predicted and correct PoS tags match (1) or not (0)
    :param category:    a string indicating the PoS being considered: only words whose predicted or correct PoS tag are
                        considered to compute statistics (to get global statistics for all the categories of interest,
                        simply include this function in a loop over all the categories of interest)
    :return tp:         the number of true positives
    :return tp_fp:      the sum of true and false positives
    :return tp_fn:      the sum of true positives and false negatives
    """

    tp = 0
    tp_fp = 0
    tp_fn = 0

    for k in hits:
        if hits[k]['predicted'] == category:
            tp_fp += 1

        if hits[k]['correct'] == category:
            tp_fn += 1

        if hits[k]['predicted'] == category and hits[k]['correct'] == category:
            tp += 1

    return tp, tp_fp, tp_fn


########################################################################################################################


def compute_category_f1(hits):

    """
    :param hits:        a dictionary of dictionaries mapping strings (the word types to be categorized) to three fields:
                        'predicted', indicating the PoS tag predicted by the model
                        'correct' indicating the gold standard PoS tag
                        'accuracy', indicating whether the two match (1) or not (0)
    :return stats:      a dictionary of dictionaries mapping each category to its corresponding recall, precision, and
                        f1 scores. Each category is a key of the dictionary and a dictionary itself, whose keys are
                        'recall', 'precision', and 'f1', mapping to the corresponding values. A further key of the
                        top-level dictionary is 'all', consisting of the same three sub-keys, mapping to recall,
                        precision, and f1 scores for the whole experiment.
    """

    stats = defaultdict(dict)

    categories = set()
    for item in hits:
        categories.add(hits[item]['predicted'])
        categories.add(hits[item]['correct'])

    all_tp = []
    all_tp_fn = []
    all_tp_fp = []

    for category in sorted(categories):
        tp, tp_fp, tp_fn = get_precision_and_recall(hits, category)
        recall = tp / tp_fn if tp_fn != 0 else 0
        precision = tp / tp_fp if tp_fp != 0 else 0
        f1 = 0 if (precision == 0 or recall == 0) else 2 * ((precision * recall) / (precision + recall))

        stats[category]['recall'] = recall
        stats[category]['precision'] = precision
        stats[category]['f1'] = f1

        all_tp.append(tp)
        all_tp_fn.append(tp_fn)
        all_tp_fp.append(tp_fp)

    total_recall = sum(all_tp) / sum(all_tp_fn)
    total_precision = sum(all_tp) / sum(all_tp_fp)
    total_f1 = 2 * ((total_precision * total_recall) / (total_precision + total_recall))

    stats['all']['recall'] = total_recall
    stats['all']['precision'] = total_precision
    stats['all']['f1'] = total_f1

    return stats
