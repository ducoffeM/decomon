import keras.ops as K
from keras.layers import Dense

from decomon.layers.layer import DecomonLayer
from decomon.types import Tensor


class DecomonDense(DecomonLayer):
    layer: Dense
    linear = True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:
        w = self.layer.kernel
        b = self.layer.bias if self.layer.use_bias else K.zeros((self.layer.units,))

        # manage tensor-multid input
        if len(self.layer.input.shape) > 2:
            shape = list(self.layer.input.shape[1:])
            dim = K.prod(self.layer.input.shape[1:])
            input_w = K.reshape(K.identity(dim), [1] + shape + shape)
            w = K.matmul(input_w, w)[0]
            shape_b = list(self.layer.input.shape[1:-1])
            dim_b = K.prod(shape_b)
            b = K.reshape(K.repeat(b[None], dim_b, axis=0), shape_b + [-1])

        """
        for dim in self.layer.input.shape[-2:0:-1]:
            # Construct a multid-tensor diagonal by blocks
            reshaped_outer_shape = (dim, dim) + w.shape
            transposed_outer_axes = (
                (0,)
                + tuple(range(2, 2 + len(b.shape)))
                + (1,)
                + tuple(range(2 + len(b.shape), len(reshaped_outer_shape)))
            )
            w = K.transpose(K.reshape(K.outer(K.identity(dim), w), reshaped_outer_shape), transposed_outer_axes)
            # repeat bias along first dimensions
            b = K.repeat(b[None], dim, axis=0)
        """

        return w, b
