import warnings
from collections import defaultdict
from context_utils.vector_spaces.maker import sort_items, make_matrix
from context_utils.vector_spaces.printer import plot_matrix
from context_utils.pos_tagging.kNN import get_nearest_indices, get_nearest_neighbors, categorize
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.metrics.pairwise import cosine_similarity as cos
    import numpy as np
    

def sklearn_experiment(training_space, training_words,
                       test_space=None, test_words=None, contexts=None,
                       pos_mapping=None, nn=1, diag_value=None, plot=''):

    """
    :param training_space:  a 2d NumPy array storing word-context co-occurrence counts derived from the training corpus
    :param training_words:  a dictionary mapping words from the training space to the corresponding row indices in the
                            training space
    :param test_space:      a 2d NumPy array storing word-context co-occurrence counts derived from the test corpus
    :param test_words:      a dictionary mapping words from the test space to the corresponding row indices in the
                            test space. If a test space is passed, test_words has to be assigned a value, otherwise the
                            function will throw an error
    :param contexts:        a dictionary mapping contexts to their column indices in the training and test spaces;
                            default is None, because this mapping is only used in the train-test setting to keep the
                            alignment between training and test spaces
    :param pos_mapping:     a dictionary mapping CHILDES PoS tags to custom, coarser tags
    :param nn:              the number of nearest neighbours to be considered when categorizing a test word
    :param diag_value:      the value to which all the cells on the main diagonal of the matrix of cosine similarities
                            between test and training vectors are set (default is 0, meaning that cells on the main
                            diagonal don't impact the nearest neighbour computation). This option makes it possible to
                            force the model to categorize a test word while ignoring the vector from the training space
                            that correspond to the same word type, thus enforcing generalization
    :param plot:            a string indicating the path where the plot showing the cosine similarity matrix is saved
                            The default is the empty string, meaning that no plot is created
    :return hits:           a dictionary mapping each word in the test set to three fields and the corresponding value:
                            'predicted' is the PoS tag that the learner predicted for a test word
                            'correct' is the correct PoS tag as found in the CHILDES corpus
                            'accuracy' is a binary value indicating if 'predicted' and 'correct' match (1) or not (0)
    """

    t = 1 if test_space is not None else 0
    w = 1 if test_words is not None else 0
    c = 1 if contexts is not None else 0
    if sum([t, w, c]) not in [0, 3]:
        raise ValueError('Unsure whether to use a leave-one-out or training-test approach! '
                         'If you want to run a leave-one-out experiment, do not provide any argument to the parameters'
                         ' test_space, test_words, and contexts. If, however, you want to perform an experiment in the'
                         ' training-test setting, provide appropriate arguments to all three parameters.')

    hits = defaultdict(dict)

    if test_space is not None:
        # use a training-test setting, where words from the test set are categorized by retrieving nearest neighbours in
        # the training set
        target_words = test_words
        words = set(training_words.keys()).union(set(test_words.keys()))

        # map every word occurring in either the training space, the test space, or both to a numerical index and get
        # an inverted mapping from indices to strings
        word_indices = sort_items(words)
        inverted_word_indices = {v: k for k, v in word_indices.items()}

        # create a training matrix and a test matrix that have as many rows as there are words in total, and the same
        # columns as the original matrices; then compute pairwise similarities between each pair of training-test words
        training_space = make_matrix(training_space, word_indices, training_words, contexts)
        test_space = make_matrix(test_space, word_indices, test_words, contexts)
        cosine_similarities = cos(training_space, test_space)

        # if so specified in the function call, set the diagonal values to the desired number
        # the idea is to 'silence' the diagonal by setting it to 0: this because the diagonal cells correspond to the
        # cosine similarity between equal types in the training and test set (e.g. dog in the training set and dog in
        # the test set). The cosine will not be 1 because the vectors of co-occurrence will differ (they have been
        # harvested in two different corpora); yet, we can expect same types to have more similar co-occurrence patterns
        # then different types. This could bias the retrieval of nearest neighbours: dog (from the training set) will be
        # retrieved as nearest neighbour of dog (from the test set). This is not a problem per se, but it can be in some
        # experimental settings: the diag-Value allows to get rid of this by force the diagonal values to 0, so that no
        # same word from training word will be retrieved as nearest neighbour for any test item
        if diag_value is not None:
            cosine_similarities[np.diag_indices_from(cosine_similarities)] = diag_value

    else:
        # use a leave-one-out setting, where words from the training set are categorized by retrieving nearest
        # neighbours from the training set, excluding the vector of the word being categorized from the pool of possible
        # neighbours
        target_words = training_words
        words = training_words
        word_indices = sort_items(words)
        inverted_word_indices = {v: k for k, v in word_indices.items()}
        cosine_similarities = cos(training_space)

        # in a leave-one-out setting, the diagonal is always set to 0 because otherwise categorization would be perfect:
        # the same vectors would be compared, resulting in a cosine similarity of 1, which will always be the maximum.
        # To avoid this, the diagonal cells are forced to 0.
        cosine_similarities[np.diag_indices_from(cosine_similarities)] = 0

    if plot:
        plot_matrix(cosine_similarities, neighbors=10, output_path=plot)

    # Use the derived cosine similarities to find which words from the training set are closer to each of the target
    # words (which words are used as targets depend on whether a test space is passed: if it is, target words are test
    # words, if it's not, target words are training words) to be able to categorize the target words. Nearest neighbors
    # are retrieved using a nearest distance approach, meaning that when two or more words from the training set are at
    # the same closest distance from a target word, they are all considered as nearest neighbors to assign a PoS tag to
    # the target word. Ties are broken by looking for the most frequent neighbour in the training set. If there is a tie
    # a word is sammpled randomly from the pool of most frequent words among the neighbours.
    for word in target_words:
        # get the column index of the test word to be categorized, and get the indices of all the rows that have a
        # cosine similarity to the word to be categorized that is at least as high as the closest distance (if k is 1,
        # otherwise get the cosine similarity value corresponding to the second closest distance (k=2), third closest
        # distance (k=3), and so on)
        c_idx = word_indices[word]
        nearest_indices = get_nearest_indices(cosine_similarities, c_idx, nn=nn)

        # get all the word strings having a high enough cosine similarity value to the word to be categorized
        nearest_neighbors = get_nearest_neighbors(nearest_indices[0], inverted_word_indices)

        # if more than one neighbour is found at the closest distance, pick the one with the highest frequency of
        # occurrence in the training set; if more than a word has the same frequency count, pick randomly
        predicted = categorize(nearest_neighbors, training_space,
                               word_indices, pos_mapping=pos_mapping)
        hits[word]['predicted'] = predicted
        hits[word]['correct'] = pos_mapping[word.split('~')[0]] if pos_mapping else word.split('~')[0]
        hits[word]['accuracy'] = 1 if hits[word]['predicted'] == hits[word]['correct'] else 0

    return hits, cosine_similarities, word_indices
