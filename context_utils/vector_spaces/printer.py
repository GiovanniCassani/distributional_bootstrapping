import operator
import numpy as np
import matplotlib.pyplot as plt


def print_vector_space(co_occurrence, row_ids, col_ids, output_file):

    """
    :param co_occurrence:   a NumPy 2d array, where each row is a word and each column is a context. Each cell contains
                            an integer specifying how many times a word and a context co-occurred in the input corpus
    :param row_ids:         a dictionary mapping strings denoting words to their row index in the co-occurrence matrix
    :param col_ids:         a dictionary mapping strings denoting contexts to their column index in the co-occurrence
                            matrix
    :param output_file:     the path to the file where the co-occurrence matrix will be written
    """

    with open(output_file, 'a+') as f:
        for word, r_idx in sorted(row_ids.items(), key=operator.itemgetter(1)):
            counts = []
            pos, w = word.split('~')
            if w != '':
                for context, c_idx in sorted(col_ids.items(), key=operator.itemgetter(1)):
                    counts.append(co_occurrence[r_idx, c_idx])
                f.write("\t".join([word, "\t".join(["\t".join([str(c) for c in counts]), pos])]))
                f.write('\n')
                
                
def plot_matrix(x, output_path='', neighbors=0):

    """
    :param x:
    :param output_path:
    :param neighbors:
    """

    r, c = x.shape
    fig, ax_array = plt.subplots(1, 1, figsize=(10, 10), sharex='col', sharey='row')

    for ax in np.ravel(ax_array):

        if neighbors:
            # if only the nearest neighbors have to be plotted, first independently sort all columns in the array of
            x_sorted = np.sort(x, axis=0)[::-1]

            # initialize a vector of the same length, loop through all columns, turn them into a set to avoid counting
            # separately all rows sharing the same value, sort them in descending order, and get the value at the
            # desired index putting it into the newly initialized vector. If the vector of values doesn't contain enough
            # values for the index to actually point to a value, take the last value in the vector. The resulting vector
            # contains the value of the desired index for all columns (or the largest possible one).
            threshold = np.zeros(c)
            for col in range(c):
                try:
                    threshold[col] = sorted(set(x_sorted[:, col]), reverse=True)[neighbors]
                except IndexError:
                    threshold[col] = sorted(set(x_sorted[:, col]), reverse=True)[-1]

            # finally set to 0 all cosine similarity values in the NumPy array that are strictly smaller than the
            # threshold value in the corresponding column and plot
            x[np.where(x < threshold)] = 0
            ax.imshow(x, aspect='auto', interpolation='nearest')

        else:
            ax.imshow(x, aspect='auto', interpolation='nearest')

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()
