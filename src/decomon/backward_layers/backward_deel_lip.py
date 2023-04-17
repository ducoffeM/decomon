from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.backend import conv2d_transpose
from tensorflow.keras.layers import Flatten, Layer
from tensorflow.python.ops import array_ops

from decomon.backward_layers.activations import get
from decomon.backward_layers.core import BackwardLayer
from decomon.backward_layers.utils import get_affine, get_ibp, get_identity_lirpa
from decomon.layers.core import DecomonLayer, ForwardMode, Option
from decomon.layers.decomon_layers import (  # add some layers to module namespace `globals()`
    DecomonGroupSort2,
    get_groupsort2_reshape,
)
from decomon.layers.utils import ClipAlpha, NonNeg, NonPos
from decomon.layers.utils_pooling import (
    get_lower_linear_hull_max,
    get_lower_linear_hull_min,
    get_upper_linear_hull_max,
    get_upper_linear_hull_min,
)
from decomon.utils import ConvexDomainType, Slope

try:
    from deel.lip.activations import GroupSort2
except ImportError:
    logger.warning(
        "Could not import GroupSort or GroupSort2 from deel.lip.activations. "
        "Please install deel-lip for being compatible with 1 Lipschitz network (see https://github.com/deel-ai/deel-lip)"
    )


class BackwardGroupSort2(BackwardLayer):
    def __init__(
        self,
        layer: Layer,
        slope: Union[str, Slope] = Slope.V_SLOPE,
        finetune: bool = False,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        self.slope = Slope(slope)
        self.finetune = finetune
        self.finetune_param: List[tf.Variable] = []
        if self.finetune:
            self.frozen_alpha = False
        self.grid_finetune: List[tf.Variable] = []
        self.frozen_grid = False

        if isinstance(layer, DecomonGroupSort2):
            self.op_reshape_in: DecomonReshape = layer.op_reshape_in
        self.data_format = layer.data_format

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "slope": self.slope,
                "finetune": self.finetune,
            }
        )
        return config

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        """
        Args:
            input_shape: list of input shape

        Returns:

        """
        if hasattr(self, "op_reshape_in"):
            return
        input_shape = input_shape[-1]
        self.op_reshape_in, _ = get_groupsort2_reshape(input_shape, self.mode, self.data_format)

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        # infer the output dimension
        inputs_ = self.op_reshape_in(inputs)

        # compute max
        w_out_u_max, b_out_u_max = get_upper_linear_hull_max(
            inputs_, mode=self.mode, convex_domain=self.convex_domain, axis=-1
        )
        w_out_l_max, b_out_l_max = get_lower_linear_hull_max(
            inputs_, mode=self.mode, convex_domain=self.convex_domain, axis=-1
        )

        # compute min
        w_out_u_min, b_out_u_min = get_upper_linear_hull_min(
            inputs_, mode=self.mode, convex_domain=self.convex_domain, axis=-1
        )
        w_out_l_min, b_out_l_min = get_lower_linear_hull_min(
            inputs_, mode=self.mode, convex_domain=self.convex_domain, axis=-1
        )

        w_u_max, b_u_max, w_l_max, b_l_max = get_diagonal_hull(w_out_u_max, b_out_u_max, w_out_l_max, b_out_l_max)
        w_u_min, b_u_min, w_l_min, b_l_min = get_diagonal_hull(w_out_u_min, b_out_u_min, w_out_l_min, b_out_l_min)

        shape_w = w_u_max.shape[1]
        shape_b = b_u_max.shape[1]
        w_u_ = K.reshape(K.concatenate([w_u_max, w_u_min], -1), (-1, shape_w, shape_w))
        w_l_ = K.reshape(K.concatenate([w_l_max, w_l_min], -1), (-1, shape_w, shape_w))
        b_u_ = K.reshape(K.concatenate([b_u_max, b_u_min], -1), (-1, shape_b))
        b_l_ = K.reshape(K.concatenate([b_l_max, b_l_min], -1), (-1, shape_b))

        return [w_u_, b_u_, w_l_, b_l_]
