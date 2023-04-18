from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Flatten, Layer

from decomon.backward_layers.core import BackwardLayer
from decomon.layers.core import ForwardMode, Option
from decomon.layers.deel_lip import DecomonGroupSort2, get_groupsort2_reshape
from decomon.layers.utils_pooling import (
    get_lower_linear_hull_max,
    get_lower_linear_hull_min,
    get_upper_linear_hull_max,
    get_upper_linear_hull_min,
)
from decomon.utils import ConvexDomainType, Slope


class BackwardGroupSort2(BackwardLayer):
    """Backward  LiRPA of Flatten"""

    def __init__(
        self,
        layer: Layer,
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

        # self.finetune = finetune
        # self.finetune_param: List[tf.Variable] = []
        print("C")

        # if self.finetune:
        #    self.frozen_alpha = False
        # self.grid_finetune: List[tf.Variable] = []
        # self.frozen_grid = False
        # print('D')

        if isinstance(layer, DecomonGroupSort2):
            self.op_reshape_in: DecomonReshape = layer.op_reshape_in
        self.data_format = layer.data_format
        print("E")

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
        self.built = True

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

        # flatten
        shape_w = np.prod(w_out_u_max.shape[1:])
        shape_b = np.prod(b_out_u_max.shape[1:])
        w_u_max_ = K.reshape(w_out_u_max, (-1, shape_w))
        w_l_max_ = K.reshape(w_out_l_max, (-1, shape_w))
        w_u_min_ = K.reshape(w_out_u_min, (-1, shape_w))
        w_l_min_ = K.reshape(w_out_l_min, (-1, shape_w))
        b_u_max = K.reshape(b_out_u_max, (-1, shape_b, 1))
        b_l_max = K.reshape(b_out_l_max, (-1, shape_b, 1))
        b_u_min = K.reshape(b_out_u_min, (-1, shape_b, 1))
        b_l_min = K.reshape(b_out_l_min, (-1, shape_b, 1))

        w_u_max = K.reshape(tf.linalg.diag(w_u_max_), (-1, shape_w, shape_w // 2, 2))
        w_l_max = K.reshape(tf.linalg.diag(w_l_max_), (-1, shape_w, shape_w // 2, 2))
        w_u_min = K.reshape(tf.linalg.diag(w_u_min_), (-1, shape_w, shape_w // 2, 2))
        w_l_min = K.reshape(tf.linalg.diag(w_l_min_), (-1, shape_w, shape_w // 2, 2))

        w_u_max = tf.reduce_sum(w_u_max, -1)
        w_l_max = tf.reduce_sum(w_l_max, -1)
        w_u_min = tf.reduce_sum(w_u_min, -1)
        w_l_min = tf.reduce_sum(w_l_min, -1)

        # w_u_max, b_u_max, w_l_max, b_l_max = get_diagonal_hull(w_out_u_max, b_out_u_max, w_out_l_max, b_out_l_max)
        # w_u_min, b_u_min, w_l_min, b_l_min = get_diagonal_hull(w_out_u_min, b_out_u_min, w_out_l_min, b_out_l_min)

        # shape_w = w_u_max.shape[1]
        # shape_b = b_u_max.shape[1]
        w_u_ = K.reshape(K.concatenate([w_u_max, w_u_min], -1), (-1, shape_w, shape_w))
        w_l_ = K.reshape(K.concatenate([w_l_max, w_l_min], -1), (-1, shape_w, shape_w))
        b_u_ = K.reshape(K.concatenate([b_u_max, b_u_min], -1), (-1, shape_b * 2))
        b_l_ = K.reshape(K.concatenate([b_l_max, b_l_min], -1), (-1, shape_b * 2))
        return [w_u_, b_u_, w_l_, b_l_]
