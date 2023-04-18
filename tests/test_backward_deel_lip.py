import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from deel.lip.activations import GroupSort, GroupSort2
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Model

from decomon.backward_layers.backward_deel_lip import BackwardGroupSort2
from decomon.backward_layers.backward_layers import BackwardFlatten, to_backward
from decomon.layers.core import ForwardMode
from decomon.layers.decomon_layers import to_decomon
from decomon.layers.deel_lip import DecomonGroupSort2
from decomon.utils import get_forward_from_mode, get_ibp_from_mode


def test_Backward_Groupsort2_multiD_box(floatx, mode, helpers):

    odd = 0
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    layer = DecomonGroupSort2(mode=mode)
    mode = ForwardMode(mode)

    if mode == ForwardMode.HYBRID:
        input_mode = inputs[2:]
        output = layer(input_mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    elif mode == ForwardMode.AFFINE:
        input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]
        output = layer(input_mode)
        z_0, _, _, _, _ = output
    elif mode == ForwardMode.IBP:
        input_mode = [inputs[3], inputs[6]]
        output = layer(input_mode)
        u_c_0, l_c_0 = output
    else:
        raise ValueError("Unknown mode.")

    # get backward layer
    layer_backward = BackwardGroupSort2(layer=layer, mode=mode)
    # if replacing by to_backward, we get the error:
    # TypeError: Can't instantiate abstract class BackwardGroupSort2 with abstract method call
    layer_backward = to_backward(layer)

    w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode)
    f_backward = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
    # import pdb; pdb.set_trace()
    output_ = f_backward(inputs_)
    w_u_, b_u_, w_l_, b_l_ = output_

    helpers.assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_, 0) * W_u + np.minimum(w_u_, 0) * W_l, 1)[:, :, None],
        b_u_ + np.sum(np.maximum(w_u_, 0) * b_u[:, :, None], 1) + np.sum(np.minimum(w_u_, 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_, 0) * W_l + np.minimum(w_l_, 0) * W_u, 1)[:, :, None],
        b_l_ + np.sum(np.maximum(w_l_, 0) * b_l[:, :, None], 1) + np.sum(np.minimum(w_l_, 0) * b_u[:, :, None], 1),
        "groupsort_{}".format(odd),
        decimal=decimal,
    )

    K.set_floatx("float32")
    K.set_epsilon(eps)
