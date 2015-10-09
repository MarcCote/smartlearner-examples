import numpy as np
from model import ConvMADE

import theano

from nose.tools import assert_true
from numpy.testing import assert_array_equal


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
