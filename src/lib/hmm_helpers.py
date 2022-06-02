import numpy as np


ALLOWED_CHARS = 'abcdefghijklmnopqrstuvwxyz., '


def id_map(keys):
    """Return a dict mapping each char in string to a unique integer.
    
    >>> id_map('abcd')
    {'a': 0,
     'b': 1,
     'c': 2,
     'd': 3}
    """
    return {value: key for key, value in dict(enumerate(keys)).items()}


def reverse_id_map(keys):
    """Return a dict mapping a unique integer to each char in string.

    >>> reverse_id_map('abcd')
    {0: 'a',
     1: 'b',
     2: 'c',
     3: 'd'}
    """
    return dict(enumerate(keys))

def create_transmat(corpus, keys=ALLOWED_CHARS):
    """Return a transition matrix based on the sequence of letters in corpus.

    Assume that each word in the corpus is delimited by a space.

    Args:
    corpus -- an iterable containing words as elements
    keys   -- all letters that are accounted for in the transition matrix
    """
    mat = np.zeros((len(keys), len(keys)), dtype=np.int32)
    key_id_map = id_map(keys)

    last_id = None
    for letter in corpus:
        curr = letter
        try:
            curr_id = key_id_map[curr]
        except KeyError:
            print(f"Skipping unrecognized char '{letter}'")
            last_id = None
            continue
        if last_id is not None:
            mat[last_id][curr_id] += 1
        last_id = curr_id

    return mat, keys

def pprint_transmat(transmat, keys):
    """Pretty-print the transition matrix."""
    h, w = transmat.shape
    print('\t', end='')
    for key in keys:
        print(f'{key} ', end='\t\t')
    print()
    for i in range(h):
        print(f'{keys[i]} ', end='\t')
        for j in range(w):
            print(f'{transmat[i][j]:.3f} ', end='\t',)
        print()

def decode_pred(pred,keys):
    out = ''
    for i in pred:
        out += keys[i]
    return out

def pred_for_hmm(pred):
    pred_for_hmm = np.zeros((len(pred), 1), dtype=np.int32)
    for i in range(len(pred)):
        pred_for_hmm[i,0] = pred[i]
    return pred_for_hmm
