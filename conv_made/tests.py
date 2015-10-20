import numpy as np
from model import ConvMADE

import theano
import theano.tensor as T

from nose.tools import assert_true, assert_almost_equal
from numpy.testing import assert_array_equal
from utils import gen_cartesian, ichunks


def test_px_sum_to_one():
    # Create images
    image_shape = (5, 5)
    nb_pixels = int(np.prod(image_shape))
    images = gen_cartesian([(0, 1)] * nb_pixels)

    # Create model
    filter_shape = (2, 2)
    nb_filters = 10

    model = ConvMADE(image_shape, filter_shape, nb_filters)
    model.W.set_value(np.ones_like(model.W.get_value()))
    model.V.set_value(np.ones_like(model.V.get_value()))
    model.activation_fct = lambda x: x  # No activation

    # Generate a mask
    mask = np.arange(np.prod(filter_shape)).reshape(filter_shape)

    X = T.matrix("input")
    use_conv_MADE = theano.function([X], model.use(X))

    for o in range(int(np.prod(filter_shape))+1):
        model.mask.set_value((mask <= o).astype(theano.config.floatX))

        probs = []
        for batch in ichunks(images, n=2**16):
            batch = np.array(batch, dtype=theano.config.floatX)
            probs_batch = np.log(use_conv_MADE(batch)).sum(axis=1)
            probs += probs_batch.tolist()

        px = np.exp(probs)
        print(px)
        from ipdb import set_trace as dbg
        dbg()
        assert_almost_equal(np.sum(px), 1)


def test_autoregressiveness():
    image_shape = (2, 2)
    filter_shape = (2, 2)
    nb_filters = 10
    model = ConvMADE(image_shape, filter_shape, nb_filters)
    model.W.set_value(np.ones_like(model.W.get_value()))
    model.V.set_value(np.ones_like(model.V.get_value()))
    model.activation_fct = lambda x: x  # No activation

    # Create a mask
    mask = np.arange(np.prod(filter_shape)).reshape(filter_shape)

    for o in range(int(np.prod(filter_shape))+1):
        model.mask.set_value((mask <= o).astype(theano.config.floatX))

        X = np.zeros(image_shape)
        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                X[i, j] = 1
                output = model.get_output(X).reshape(image_shape).eval()
                assert_true(np.all(output[:i, :j] == 0))
                X[i, j] = 0
