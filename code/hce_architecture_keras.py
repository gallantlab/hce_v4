"""
Hierarchical Convolutional Energy Model architecture.
Implemented in Keras.
Original code by Michael Oliver. Adapted by Michele Winter 2023.
"""

import sys

from keras.initializers import uniform, constant
from keras.models import Sequential
from keras.layers import Dense, Flatten
from kfs.layers.core import FilterDims
from kfs.layers.convolutional import (
    Convolution2DEnergy_TemporalBasis,
    Convolution2DEnergy_Scatter,
)
from kfs.layers.noise import CoupledGaussianDropout, Gain, AxesDropout
from kfs.regularizers import LaplacianRegularizer, TVRegularizer
from kfs.layers.advanced_activations import ParametricSoftplus
from keras_contrib.optimizers import ftml

delays = range(2, 16)
u1 = uniform(minval=-0.01, maxval=0.01)


def create_model(verbose=True):
    """
    Create HCE model.
    """
    if verbose:
        print("Creating model")
    sys.stdout.flush()
    tfilt = 3
    tfeat = 5
    sfilt = 10
    f1s = 8
    f1c = 8
    f2s = 8
    f2c = 8
    gain = 0.01
    fz = 7
    n = 0.5

    model = Sequential()

    model.add(
        Convolution2DEnergy_TemporalBasis(
            f1s,
            f1c,
            tfilt,
            (fz, fz),
            tfeat,
            padding="valid",
            strides=(2, 2),
            temporal_frequencies_initial_max=3,
            temporal_frequencies_scaling=5,
            temporal_kernel_regularizer=LaplacianRegularizer(l2=0.00002, axis=0),
            input_shape=(len(delays), 3, 64, 64),
        )
    )
    model.add(CoupledGaussianDropout(n))
    model.add(
        FilterDims(
            20,
            sum_axes=(1, 2),
            filter_axes=(1, 2),
            activation=None,
            use_bias=False,
            kernel_initializer=u1,
        )
    )
    model.add(
        Convolution2DEnergy_Scatter(f2s, f2c, (fz, fz), padding="valid", strides=(2, 2))
    )
    model.add(CoupledGaussianDropout(n))
    model.add(
        FilterDims(
            20,
            sum_axes=(1,),
            filter_axes=(1,),
            activation=None,
            use_bias=False,
            kernel_initializer=u1,
        )
    )
    model.add(
        FilterDims(
            sfilt,
            sum_axes=(2, 3),
            filter_axes=(2, 3),
            activation=None,
            kernel_regularizer=TVRegularizer(0.0, 0.00002, axes=[1, 2]),
            use_bias=False,
            kernel_initializer=u1,
        )
    )
    model.add(Flatten())
    model.add(CoupledGaussianDropout(n))
    model.add(AxesDropout(0.5, axes=(1,)))
    model.add(Dense(10, kernel_initializer=u1, activation="relu"))
    model.add(CoupledGaussianDropout(n))
    model.add(Dense(1, kernel_initializer=u1))
    model.add(Gain(gain))
    model.add(
        ParametricSoftplus(
            alpha_initializer=constant(0.1),
            beta_initializer=constant(10.0),
            trainable=False,
        )
    )

    opt = ftml(lr=0.0005)
    model.compile(loss="poisson", optimizer=opt)
    return model
