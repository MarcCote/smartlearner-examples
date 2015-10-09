from __future__ import division

import theano.tensor as T
from theano.tensor.nnet import conv
import smartlearner.initializers as initer

import numpy as np

from smartlearner.interfaces import Model
from smartlearner.utils import sharedX


class ConvMADE(Model):
    def __init__(self, image_shape, filter_shape, nb_filters=10):
        self.image_shape = image_shape
        self.filter_shape = filter_shape

        self.activation_fct = T.nnet.sigmoid

        # Allocating memory for parameters
        nb_input_feature_maps = 1
        #nb_filters = int(np.prod(filter_shape))
        #W_shape = (nb_filters, nb_input_feature_maps) + filter_shape
        W_shape = (nb_filters, nb_input_feature_maps) + filter_shape
        self.W = sharedX(value=np.zeros(W_shape), name='W', borrow=True)
        self.b = sharedX(value=np.zeros(nb_filters), name='b', borrow=True)

        V_shape = (1, nb_filters) + filter_shape
        self.V = sharedX(value=np.zeros(V_shape), name='V', borrow=True)
        self.c = sharedX(value=np.zeros(1), name='c', borrow=True)

        # Build a temporary masks (a new one should be sampled each update).
        M = np.arange(np.prod(self.filter_shape)).reshape(self.filter_shape)
        M = M < 1
        self.mask = sharedX(value=M, name='M', borrow=True)

    def initialize(self, weights_initializer=initer.UniformInitializer(random_seed=1234)):
        weights_initializer(self.W)
        weights_initializer(self.V)

    @property
    def updates(self):
        return {}  # No updates.

    @property
    def parameters(self):
        return [self.W, self.b, self.V, self.c]

    def get_output(self, X):
        # Hack: X is a 2D matrix instead of a 4D tensor, but we have all the information to fix that.
        nb_input_feature_maps = 1  # One channel
        X = X.reshape((-1, nb_input_feature_maps) + self.image_shape)

        mask = self.mask.dimshuffle('x', 'x', 0, 1)
        # Filters are flip? See theano doc on conv.conv2d.
        mask = mask[:, :, ::-1, ::-1]

        layer1_conv = conv.conv2d(X, filters=self.W*mask, border_mode="valid")
        layer1_pre_output = layer1_conv + self.b.dimshuffle('x', 0, 'x', 'x')
        layer1 = self.activation_fct(layer1_pre_output)

        mask_inverse_and_flip = 1-mask[:, :, ::-1, ::-1]
        model_conv = conv.conv2d(layer1, filters=self.V*mask_inverse_and_flip, border_mode="full")
        model_pre_output = model_conv + self.c.dimshuffle('x', 0, 'x', 'x')
        model_out = self.activation_fct(model_pre_output)
        return model_out.reshape((X.shape[0], -1))

        #conv_out = conv.conv2d(X, filters=self.W*self.M, border_mode="full")
        #conv_out = conv.conv2d(X, filters=self.W*self.M, border_mode="valid", subsample=self.filter_shape)
        #conv_out = conv.conv2d(conv_out, filters=1-self.M, border_mode="full")

        # Keep only non overlapping patches.
        # Crop fillvalues that have been added.
        #conv_out = conv_out[:, :, 1::self.filter_shape[0], 1::self.filter_shape[1]]

        #return pre_output.dimshuffle(0, 2, 3, 1).reshape((-1, T.prod(pre_output.shape[1:])))
        #return output.dimshuffle(0, 2, 3, 1).reshape((-1, T.prod(output.shape[1:])))

    def use(self, X):
        #probs = T.nnet.sigmoid(self.get_output(X))
        probs = self.get_output(X)
        return probs

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass
