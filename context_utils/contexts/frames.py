import operator
from collections import Counter
from context_utils.readers import read_txt_corpus_file
from context_utils.utterance import clean_utterance


def collect_frames(input_corpus, pos_dict=None, boundaries=True, freq_frames=True, flex_frames=False):

    """
    :param input_corpus:    the path to a .txt file containing CHILDES transcripts, with one utterance per line and
                            words divided by white spaces. The first element of each utterance is the capitalized label
                            of the speaker, as found in CHILDES. The second element is a dummy word marking the
                            beginning of the utterance, #start; the last element is a dummy word marking the end of the
                            utterance, #end. Each word is paired to its Part-of-Speech tag, the two separated by a
                            tilde, word~PoS.
    :param pos_dict:        a dictionary mapping CHILDES PoS tags to custom tags; any CHILDES tag that doesn't appear as
                            key in the dictionary is assumed to be irrelevant and words labeled with those tags are
                            discarded; the default to None means that every input word is considered
    :param boundaries:      a boolean indicating whether utterance boundaries are to be considered or not as context
                            elements.
    :param freq_frames:     a boolean indicating whether frequent frames are collected
    :param flex_frames:     a boolean indicating whether flexible frames are collected
    :return frames:         a dictionary containing the n most frequent frames as keys and the corresponding frequency
                            counts as values; words within a frame are separated by two underscores ('__')

    The function makes it possible to harvest frequent and flexible frames at the same time, but the two are stored in
    the same dictionary: if both parameters are set to True, a way must be devised to discriminate frequent and flexible
    frames afterwards: it's fairly trivial given the structure of the frames, but it's not implemented here. Ideally,
    this function is called with either of frequent and flexible frames, but not both.
    """

    frames = Counter()
    corpus = read_txt_corpus_file(input_corpus)

    sep = '__'

    for line in corpus:
        if line[0] != 'CHI':
            # get rid of the speaker ID once made sure the utterance doesn't come from the child
            del line[0]

            # get rid of unwanted elements from the original utterance
            words = clean_utterance(line, boundaries=boundaries, pos_dict=pos_dict)

            # get the numerical index of the last element in the utterance. If frequent frames are to be collected,
            # subtract one from the last index and add one to the first index, because no trigram can be collected
            # for the first and last elements
            if flex_frames:
                last_idx = len(words)
                idx = 0
                while idx < last_idx:
                    # collect flexible frames, minding that no right context is collected for the end of utterance
                    # dummy word and no left context is collected for the beginning of utterance dummy word; store
                    # the frequency count of each frame
                    if words[idx].split('~', 1)[1] != '#end':
                        frame = sep.join([words[idx].split('~', 1)[1], 'X'])
                        frames[frame] += 1
                    if words[idx].split('~', 1)[1] != '#start':
                        frame = sep.join(['X', words[idx].split('~', 1)[1]])
                        frames[frame] += 1
                    idx += 1

            if freq_frames:
                last_idx = len(words) - 1
                idx = 1
                while idx < last_idx:
                    # collect flexible frames
                    frame = sep.join([words[idx-1].split('~', 1)[1],
                                      'X', words[idx+1].split('~', 1)[1]])
                    frames[frame] += 1

                    idx += 1

    return frames


########################################################################################################################


def get_salient_frames(frames, n):

    """
    :param frames:  a dictionary mapping strings to integers
    :param n:       the number of elements from the input dictionary to be returned
    :return:        a dictionary mapping strings to integers, containing the top n elements (sorted by value in
                    descending order) from the input dictionary
    """

    output_frames = {}
    c = 1
    for k in sorted(frames.items(), key=operator.itemgetter(1), reverse=True):
        if c <= n:
            output_frames[k[0]] = frames[k[0]]
            c += 1
        else:
            return output_frames
