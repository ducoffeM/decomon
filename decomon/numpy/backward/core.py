from __future__ import absolute_import
from decomon.utils import F_FORWARD
"""
CROWN implementation in numpy (useful for MILP compatibility)
"""
class BackwardNumpyLayer(object):

    def __init__(self, keras_layer, convex_domain={}, mode=F_FORWARD.name, rec=1, params=[],  **kwargs):
        """

        :param convex_domain: type of convex input domain (None or dict)
        :param dc_decomp: boolean that indicates whether we return a
        difference of convex decomposition of our layer
        :param mode: type of Forward propagation (IBP, Forward or Hybrid)
        :param kwargs: extra parameters
        """
        self.update_slope=False
        self.reuse_slope=False
        if 'update_slope' in kwargs:
            self.update_slope= kwargs['update_slope']
        if 'reuse_slope' in kwargs:
            self.reuse_slope=kwargs['reuse_slope']

        kwargs.pop('update_slope')
        kwargs.pop('reuse_slope')
        super(BackwardNumpyLayer, self).__init__(**kwargs)
        if mode != F_FORWARD.name:
            raise NotImplementedError()
        self.keras_layer = keras_layer
        self.convex_domain = convex_domain
        self.rec=rec
        self.mode = mode
        self.params=params


    def call(self, inputs, **kwargs):

        pass

    def store_layer(self, joint=False, convex_domain={}, reuse_slope=False):
        return False

    def store_output(self, joint=False, convex_domain={}, reuse_slope=False):
        return True

    def update(self, x, params):
        pass