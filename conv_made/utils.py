import os
import sys
import numpy as np
from time import time

import theano
import itertools

from smartlearner.interfaces import Dataset

DATASETS_ENV = 'DATASETS'


class Timer():
    def __init__(self, txt):
        self.txt = txt

    def __enter__(self):
        self.start = time()
        print(self.txt + "... ", end="")
        sys.stdout.flush()

    def __exit__(self, type, value, tb):
        print("{:.2f} sec.".format(time()-self.start))

DATASETS_ENV = 'DATASETS'


def load_binarized_mnist():
    #Temporary patch until we build the dataset manager
    dataset_name = "binarized_mnist"

    datasets_repo = os.environ.get(DATASETS_ENV, './datasets')
    if not os.path.isdir(datasets_repo):
        os.mkdir(datasets_repo)

    repo = os.path.join(datasets_repo, dataset_name)
    dataset_npy = os.path.join(repo, 'data.npz')

    if not os.path.isfile(dataset_npy):
        if not os.path.isdir(repo):
            os.mkdir(repo)

            import urllib
            urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_train.txt', os.path.join(repo, 'mnist_train.txt'))
            urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_valid.txt', os.path.join(repo, 'mnist_valid.txt'))
            urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_test.txt', os.path.join(repo, 'mnist_test.txt'))

        train_file, valid_file, test_file = [os.path.join(repo, 'mnist_' + ds + '.txt') for ds in ['train', 'valid', 'test']]
        rng = np.random.RandomState(42)

        def parse_file(filename):
            data = np.array([np.fromstring(l, dtype=np.float32, sep=" ") for l in open(filename)])
            data = data[:, :-1]  # Remove target
            data = (data > rng.rand(*data.shape)).astype('int8')
            return data

        trainset, validset, testset = parse_file(train_file), parse_file(valid_file), parse_file(test_file)
        np.savez(dataset_npy,
                 trainset_inputs=trainset,
                 validset_inputs=validset,
                 testset_inputs=testset)

    data = np.load(dataset_npy)
    trainset = Dataset(data['trainset_inputs'].astype(theano.config.floatX), name="trainset")
    validset = Dataset(data['validset_inputs'].astype(theano.config.floatX), name="validset")
    testset = Dataset(data['testset_inputs'].astype(theano.config.floatX), name="testset")

    return trainset, validset, testset


def gen_cartesian(sequences):
    """
    Creates a generator of cartesian products form by input arrays.

    Parameters
    ----------
    sequences : list of array-like
        1-D arrays to form the cartesian product of.

    Returns
    -------
    out : generator
        generator producing 1D array of shape (1, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> list(cartesian(([1, 2, 3], [4, 5], [6, 7])))
    [array([1, 4, 6]),
     array([1, 4, 7]),
     array([1, 5, 6]),
     array([1, 5, 7]),
     array([2, 4, 6]),
     array([2, 4, 7]),
     array([2, 5, 6]),
     array([2, 5, 7]),
     array([3, 4, 6]),
     array([3, 4, 7]),
     array([3, 5, 6]),
     array([3, 5, 7])]

    """

    return itertools.product(*sequences)


def chunks(sequence, n):
    """ Yield successive n-sized chunks from sequence.
    """
    for i in xrange(0, len(sequence), n):
        yield sequence[i:i+n]


def ichunks(iterable, n):
    """ Yield successive n-sized chunks from sequence.
    """
    while True:
        yield list(itertools.islice(iterable, n))
