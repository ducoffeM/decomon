from typing import Any, Dict, List, Optional, Tuple, Type, Union

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec, Layer, Permute, Reshape

from decomon.core import PerturbationDomain
from decomon.layers.core import DecomonLayer, ForwardMode


class DecomonReshape(DecomonLayer, Reshape):
    """Forward LiRPA implementation of Reshape layers.
    See Keras official documentation for further details on the Reshape operator
    """

    original_keras_layer_class = Reshape

    def __init__(
        self,
        target_shape: Tuple[int, ...],
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            data_format
            **kwargs
        """
        super().__init__(
            target_shape=target_shape,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

        if self.mode == ForwardMode.HYBRID:
            self.input_spec = [
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # u_c
                InputSpec(min_ndim=1),  # w_u
                InputSpec(min_ndim=1),  # b_u
                InputSpec(min_ndim=1),  # l_c
                InputSpec(min_ndim=1),  # w_l
                InputSpec(min_ndim=1),  # b_l
            ]
        elif self.mode == ForwardMode.IBP:
            self.input_spec = [
                InputSpec(min_ndim=1),  # u_c
                InputSpec(min_ndim=1),  # l_c
            ]
        if self.mode == ForwardMode.AFFINE:
            self.input_spec = [
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # w_u
                InputSpec(min_ndim=1),  # b_u
                InputSpec(min_ndim=1),  # w_l
                InputSpec(min_ndim=1),  # b_l
            ]

        if self.dc_decomp:
            self.input_spec += [InputSpec(min_ndim=1), InputSpec(min_ndim=1)]

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        def op(x: tf.Tensor) -> tf.Tensor:
            return Reshape.call(self, x)

        nb_tensors = self.nb_tensors
        if self.dc_decomp:
            h, g = inputs[-2:]
            h_out = op(h)
            g_out = op(g)
            nb_tensors -= 2

        if self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensors]
        elif self.mode == ForwardMode.IBP:
            u_c, l_c = inputs[:nb_tensors]
        elif self.mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensors]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            u_c_out = op(u_c)
            l_c_out = op(l_c)

        if self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
            b_u_out = op(b_u)
            b_l_out = op(b_l)

            if len(w_u.shape) == len(b_u.shape):
                w_u_out = op(w_u)
                w_l_out = op(w_l)

            else:

                def step_func(x: tf.Tensor, _: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
                    return op(x), _

                w_u_out = K.rnn(step_function=step_func, inputs=w_u, initial_states=[], unroll=False)[1]
                w_l_out = K.rnn(step_function=step_func, inputs=w_l, initial_states=[], unroll=False)[1]

        if self.mode == ForwardMode.HYBRID:
            output = [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
        elif self.mode == ForwardMode.AFFINE:
            output = [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
        elif self.mode == ForwardMode.IBP:
            output = [u_c_out, l_c_out]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            output += [h_out, g_out]

        return output


class DecomonPermute(DecomonLayer, Permute):
    """Forward LiRPA implementation of Reshape layers.
    See Keras official documentation for further details on the Reshape operator
    """

    original_keras_layer_class = Permute

    def __init__(
        self,
        dims: Tuple[int, ...],
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            data_format
            **kwargs
        """
        super().__init__(
            dims=dims,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

        if self.mode == ForwardMode.HYBRID:
            self.input_spec = [
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # u_c
                InputSpec(min_ndim=1),  # w_u
                InputSpec(min_ndim=1),  # b_u
                InputSpec(min_ndim=1),  # l_c
                InputSpec(min_ndim=1),  # w_l
                InputSpec(min_ndim=1),  # b_l
            ]
        elif self.mode == ForwardMode.IBP:
            self.input_spec = [
                InputSpec(min_ndim=1),  # u_c
                InputSpec(min_ndim=1),  # l_c
            ]
        elif self.mode == ForwardMode.AFFINE:
            self.input_spec = [
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # w_u
                InputSpec(min_ndim=1),  # b_u
                InputSpec(min_ndim=1),  # w_l
                InputSpec(min_ndim=1),  # b_l
            ]
        else:
            raise ValueError(f"Unknown mode {mode}")

        if self.dc_decomp:
            self.input_spec += [InputSpec(min_ndim=1), InputSpec(min_ndim=1)]

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        def op(x: tf.Tensor) -> tf.Tensor:
            return Permute.call(self, x)

        nb_tensors = self.nb_tensors
        if self.dc_decomp:
            h, g = inputs[-2:]
            h_out = op(h)
            g_out = op(g)
            nb_tensors -= 2

        if self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensors]
        elif self.mode == ForwardMode.IBP:
            u_c, l_c = inputs[:nb_tensors]
        elif self.mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensors]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            u_c_out = op(u_c)
            l_c_out = op(l_c)

        if self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
            b_u_out = op(b_u)
            b_l_out = op(b_l)

            if len(w_u.shape) == len(b_u.shape):
                w_u_out = op(w_u)
                w_l_out = op(w_l)
            else:

                def step_func(x: tf.Tensor, _: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
                    return op(x), _

                w_u_out = K.rnn(step_function=step_func, inputs=w_u, initial_states=[], unroll=False)[1]
                w_l_out = K.rnn(step_function=step_func, inputs=w_l, initial_states=[], unroll=False)[1]

        if self.mode == ForwardMode.HYBRID:
            output = [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
        elif self.mode == ForwardMode.AFFINE:
            output = [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
        elif self.mode == ForwardMode.IBP:
            output = [u_c_out, l_c_out]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            output += [h_out, g_out]

        return output
