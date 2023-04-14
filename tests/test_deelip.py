import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from deel.lip.activations import GroupSort, GroupSort2
from numpy.testing import assert_allclose, assert_almost_equal

from decomon.layers.core import ForwardMode
from decomon.layers.deel_lip import DecomonGroupSort2


def test_groupsort2(axis, mode, floatx, helpers):

    odd = 0  # only working for multiple of 2
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-2)
    if floatx == 16:
        decimal = 2
    else:
        decimal = 5

    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)

    x, y, z, u_c, _, b_u, l_c, _, b_l = inputs

    x_ = inputs_[0]
    z_ = inputs_[2]

    layer = DecomonGroupSort2(mode=mode)
    layer_ref = GroupSort2()

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = layer(inputs[2:])
    elif mode == ForwardMode.AFFINE:
        output = layer([z, W_u, b_u, W_l, b_l])
    elif mode == ForwardMode.IBP:
        output = layer([u_c, l_c])
    else:
        raise ValueError("Unknown mode.")

    f_ref = K.function(inputs, layer_ref(inputs[1]))
    f_decomon = K.function(inputs, output)

    y_ = f_ref(inputs_)

    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_decomon(inputs_)
        helpers.assert_output_properties_box(
            x_,
            y_,
            None,
            None,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            w_u_,
            b_u_,
            l_c_,
            w_l_,
            b_l_,
            "max_multid_{}".format(odd),
            decimal=decimal,
        )

    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_ = f_decomon(inputs_)
        helpers.assert_output_properties_box(
            x_,
            y_,
            None,
            None,
            z_[:, 0],
            z_[:, 1],
            None,
            w_u_,
            b_u_,
            None,
            w_l_,
            b_l_,
            "max_multid_{}".format(odd),
            decimal=decimal,
        )

    elif mode == ForwardMode.IBP:
        u_c_, l_c_ = f_decomon(inputs_)
        helpers.assert_output_properties_box(
            x_,
            y_,
            None,
            None,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "decomon_multid_{}".format(odd),
            decimal=decimal,
        )
    else:
        raise ValueError("Unknown mode.")

    K.set_epsilon(eps)
    K.set_floatx("float{}".format(32))
