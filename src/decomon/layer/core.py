from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer, Flatten

from decomon.core import (
    BoxDomain,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    get_affine,
    get_ibp,
)
from decomon.keras_utils import get_weight_index_from_name

class Propagation(str, Enum):
    ForwardPropagation = "forward-propagation"
    BackwardPropagation = "backward-propagation"

class DecomonLayer(ABC, Layer):
    """Abstract class that contains the common information of every implemented layers for Forward LiRPA"""

    _trainable_weights: List[tf.Variable]

    @property
    def original_keras_layer_class(self) -> Type[Layer]:
        """The keras layer class from which this class is the decomon equivalent."""
        return self.layer

    def __init__(
        self,
        layer: Layer
        perturbation_domain: Optional[PerturbationDomain] = None,
        mode_in: Union[str, ForwardMode] = ForwardMode.HYBRID,
        mode_out: Union[str, ForwardMode] = ForwardMode.HYBRID,
        propagation: Union[ForwardPropagation, BackwardPropagation] = ForwardPropagation,
        finetune: bool = False,
        use_inputs: bool = True,
        has_backward: bool = True
        **kwargs: Any,
    ):
        """
        Args:
            layer: Keras Layer
            perturbation_domain: type of convex input domain (None or dict)
            mode_in: type of Forward propagation at the input (ibp, affine, or hybrid)
            mode_out: type of Forward propagation at the output (ibp, affine, or hybrid)
            propagation: direction of bounds propagation (forward: from input to output; backward: from output to input)
            finetune: boolean whether or not we introduce extra variables for optimising the bounds
            use_inputs ...
            **kwargs: extra parameters
        """
        super().__init__(**kwargs)
        if perturbation_domain is None:
            perturbation_domain = BoxDomain()
        self.inputs_outputs_spec = InputsOutputsSpec(
            dc_decomp=False, mode=mode, perturbation_domain=perturbation_domain
        )
        self.layer = layer
        self.perturbation_domain = perturbation_domain
        self.mode_in = ForwardMode(mode_in)
        self.mode_out = ForwardMode(mode_out)
        self.finetune = finetune  # extra optimization with hyperparameters
        self.propagation = propagation
        self.has_backward = has_backward
        

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "finetune": self.finetune,
                "mode": self.mode,
                "perturbation_domain": self.perturbation_domain,
                "layer": self.layer.get_config()
            }
        )
        return config

    @abstractmethod
    def get_affine_relaxations(self, upper: Union[None, tf.Tensor], lower: Union[None, tf.Tensor])-> List[tf.Tensor]:
        """_summary_

        Args:
            upper: Tensor of upper bounds inputs of the layer
            lower: Tensor of lower bounds inputs of the layer
        List of affine relaxations of the layer: w_u, b_u, w_l, b_l
        """
        pass

    @abstractmethod
    def get_ibp_relaxations(self, upper: tf.Tensor, lower: tf.Tensor)-> List[tf.Tensor]:
        """_summary_

        Args:
            upper: Tensor of upper bounds inputs of the layer
            lower: Tensor of lower bounds inputs of the layer
        List of constants bounds of the output of the layer: upper, lower
        """
        pass

    def call(self, inputs: Union[None, List[tf.Tensor]], x: Union[None, tf.Tensor], backward_inputs: Union[None, List[tf.Tensor]], **kwargs: Any) -> List[tf.Tensor]:
        """
        Args:
            inputs

        Returns:

        """

        upper: Union[None, tf.Tensor]
        lower: Union[None, tf.Tensor]
        u: tf.Tensor
        l: tf.Tensor
        affine_layer_bounds: List[tf.Tensor]

        if (get_affine(self.mode_out) and self.use_inputs) or get_ibp(self.mode_out):
            upper = self.perturbation_domain.get_upper(x, inputs, self.mode_in)
            lower = self.perturbation_domain.get_lower(x, inputs, self.mode_in)

        if get_ibp(self.mode_out):
            u, l = self.get_ibp_relaxations(upper, lower)

        if get_affine(self.mode_out):
            if not self.use_inputs:
                upper, lower = None, None
            affine_layer_bounds = self.get_affine_relaxations(upper, lower)


        if self.propagation.value == Propagation.ForwardPropagation.value and get_affine(self.mode_out):
            affine_inputs = get_affine_bounds(inputs, self.mode_in)
            output = merge_affine_bounds(affine_inputs, affine_bounds)

            if get_ibp(self.mode_out):
                upper_affine = self.perturbation_domain.get_upper(x, output, ForwardMode.FORWARD)
                lower_affine = self.perturbation_domain.get_lower(x, output, ForwardMode.FORWARD)

                u, l = K.minimum(u, upper_affine), K.maximum(l, lower_affine)
                output = fuse(output, [u, l])

        if self.propagation.value == Propagation.BackwardPropagation.value:
            if has_backward:
                output = merge_affine_bounds(affine_bounds, backward_inputs)
            else:
                output = affine_bounds

        return output

def fuse(affine_bounds, constant_bounds):
    return [constant_bounds[0]]+ affine_bounds[:2]+ [constant_bounds[1]]+ affine_bounds[-2:]


def merge_affine_bounds(w_0: tf.Tensor, b_0: tf.Tensor, w_1: tf.Tensor, b_1: tf.Tensor, shape_0: List[tf.TensorShape], shape_1: List[tf.TensorShape], shape_x: List[tf.TensorShape], op_reshape_0: Reshape, op_reshape_1: Reshape)-> List[tf.Tensor]:
    """_summary_
       n_in = prod(shape_0)
       n_out = prod(shape_1)
       Combine successive affine transformations:  first an affine transormation with respective weights and bias w_0, b_0, followed by another affine transformation with respective weights and bias w_1, b_1
       w_1*(w_0*z + b_0) + b_1 = (w_1*w_0)*z + w_1*b_0 + b_1
       w_10 = (w_1*w_0)
       b_10 = w_1*b_0
       b_11 = b_10 + b_1

    Args:
        inputs_0 (_type_): _description_
        inputs_1 (_type_): _description_

        op_reshape_0 = Reshape((prod(shape_0), -1))
        op_reshape_1 = Reshape((prod(shape_1), -1))



        
    """

    shape_w_0 = len(w_0.shape[1:])
    shape_w_1 = len(w_1.shape[1:])

    is_diag_0: bool = False
    is_diag_1: bool = False
    # computation time ???
    if shape_w_0 == len(shape_0):
        # diagonal layer
        is_diag_0 = True
    if shape_w_1 == len(shape_1):
        # diagonal layer
        is_diag_1 = True

    if is_diag_0 and is_diag_1:
        # do something
    elif is_diag_0:
        # do something else
    elif is_diag_1:
        # do something else else
    else:
        # standart dot product

        # flatten w_0
        w_0_flat = op_reshape_0(w_0) # (None, n_in, n_h)
        w_1_flat = op_reshape_1(w_1) # (None, n_h, n_out)
        b_0_flat = op_reshape_1(b_0) # (None, n_h, 1)

        w_10_flat = K.batch_dot(w_0_flat, w_1_flat, (-1, -2)) # (None, n_in, n_out)
        b_10_flat = K.batch_dot(w_1_flat, b_0_flat, (-2, -2))[:, :, 0] # (None, n_out)
        b_11_flat = b_10_flat + Flatten()(b_1) # (None, n_out)

        # reshape to the right shape
        n_dim_h = len(shape_1)
        shape_out = w_1.shape[1+n_dim_h:]
        output_shape_w = shape_x + shape_out # attention to the right (list or tuple ???)
        w_10 = K.reshape(w_10_flat, [-1]+output_shape_w)
        b_11 = K.reshape(b_11_flat, [-1]+output_shape_w)

        return [w_10, b_11]













    




