import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras.layers import Layer

from decomon.core import ForwardMode, PerturbationDomain
from decomon.layers.core import DecomonLayer
from decomon.layers.decomon_merge_layers import DecomonConcatenate
from decomon.layers.decomon_reshape import DecomonReshape
from decomon.layers.utils import ClipAlpha, expand_dims, max_, min_, sort

logger = logging.getLogger(__name__)

try:
    from deel.lip.activations import GroupSort, GroupSort2
except ImportError:
    logger.warning(
        "Could not import GroupSort or GroupSort2 from deel.lip.activations. "
        "Please install deel-lip for being compatible with 1 Lipschitz network (see https://github.com/deel-ai/deel-lip)"
    )
else:

    class DecomonGroupSort(DecomonLayer):
        original_keras_layer_class = GroupSort

        def __init__(
            self,
            n: Optional[int] = None,
            data_format: str = "channels_last",
            k_coef_lip: float = 1.0,
            perturbation_domain: Optional[PerturbationDomain] = None,
            dc_decomp: bool = False,
            mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
            finetune: bool = False,
            shared: bool = False,
            fast: bool = True,
            **kwargs: Any,
        ):
            super().__init__(
                perturbation_domain=perturbation_domain,
                dc_decomp=dc_decomp,
                mode=mode,
                finetune=finetune,
                shared=shared,
                fast=fast,
                **kwargs,
            )
            self.data_format = data_format
            if data_format == "channels_last":
                self.channel_axis = -1
            elif data_format == "channels_first":
                raise RuntimeError("channels_first not implemented for GroupSort activation")
            else:
                raise RuntimeError("data format not understood")
            self.n = n
            self.concat = DecomonConcatenate(
                mode=self.mode, perturbation_domain=self.perturbation_domain, dc_decomp=self.dc_decomp
            ).call

        def get_config(self) -> Dict[str, Any]:
            config = super().get_config()
            config.update(
                {
                    "data_format": self.data_format,
                    "mode": self.mode,
                    "n": self.n,
                }
            )
            return config

        def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
            channel_dim = input_shape[-1][self.channel_axis]
            if channel_dim is None:
                raise ValueError(f"Dimension {self.channel_axis} corresponding to `channel_axis` cannot be None")

            if (self.n is None) or (self.n > channel_dim):
                self.n = channel_dim
                if self.n is None:
                    raise RuntimeError("self.n cannot be None at this point.")
            self.reshape = DecomonReshape(
                (-1, self.n), mode=self.mode, perturbation_domain=self.perturbation_domain, dc_decomp=self.dc_decomp
            ).call

        def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
            shape_in = tuple(inputs[-1].shape[1:])
            inputs_reshaped = self.reshape(inputs)
            if self.n == 2:
                output_max = expand_dims(
                    max_(
                        inputs_reshaped,
                        dc_decomp=self.dc_decomp,
                        perturbation_domain=self.perturbation_domain,
                        mode=self.mode,
                        axis=-1,
                    ),
                    dc_decomp=self.dc_decomp,
                    mode=self.mode,
                    axis=-1,
                    perturbation_domain=self.perturbation_domain,
                )
                output_min = expand_dims(
                    min_(
                        inputs_reshaped,
                        dc_decomp=self.dc_decomp,
                        perturbation_domain=self.perturbation_domain,
                        mode=self.mode,
                        axis=-1,
                    ),
                    dc_decomp=self.dc_decomp,
                    mode=self.mode,
                    axis=-1,
                    perturbation_domain=self.perturbation_domain,
                )
                outputs = self.concat(output_min + output_max)

            else:
                outputs = sort(
                    inputs_reshaped,
                    axis=-1,
                    dc_decomp=self.dc_decomp,
                    perturbation_domain=self.perturbation_domain,
                    mode=self.mode,
                )

            return DecomonReshape(
                shape_in, mode=self.mode, perturbation_domain=self.perturbation_domain, dc_decomp=self.dc_decomp
            ).call(outputs)

    class DecomonGroupSort2(DecomonLayer):
        original_keras_layer_class = GroupSort2

        def __init__(
            self,
            n: int = 2,
            data_format: str = "channels_last",
            k_coef_lip: float = 1.0,
            perturbation_domain: Optional[PerturbationDomain] = None,
            dc_decomp: bool = False,
            mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
            finetune: bool = False,
            shared: bool = False,
            fast: bool = True,
            **kwargs: Any,
        ):
            super().__init__(
                perturbation_domain=perturbation_domain,
                dc_decomp=dc_decomp,
                mode=mode,
                finetune=finetune,
                shared=shared,
                fast=fast,
                **kwargs,
            )
            self.data_format = data_format

            if self.data_format == "channels_last":
                self.axis = -1
            else:
                self.axis = 1

            if self.dc_decomp:
                raise NotImplementedError()

            self.op_concat = DecomonConcatenate(self.axis, mode=self.mode, perturbation_domain=self.perturbation_domain)

        def get_config(self) -> Dict[str, Any]:
            config = super().get_config()
            config.update(
                {
                    "data_format": self.data_format,
                    "mode": self.mode,
                }
            )
            return config

        def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
            inputs_reshaped = self.op_reshape_in(inputs)
            inputs_max = expand_dims(
                max_(
                    inputs_reshaped,
                    mode=self.mode,
                    perturbation_domain=self.perturbation_domain,
                    axis=self.axis,
                    finetune=self.finetune,
                    finetune_params=self.params_max,
                ),
                mode=self.mode,
                axis=self.axis,
                perturbation_domain=self.perturbation_domain,
            )
            inputs_min = expand_dims(
                min_(
                    inputs_reshaped,
                    mode=self.mode,
                    perturbation_domain=self.perturbation_domain,
                    axis=self.axis,
                    finetune=self.finetune,
                    finetune_params=self.params_min,
                ),
                mode=self.mode,
                axis=self.axis,
                perturbation_domain=self.perturbation_domain,
            )
            output = self.op_concat(inputs_min + inputs_max)
            return self.op_reshape_out(output)

        def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
            single_input_shape = input_shape[-1]
            single_input_shape_wo_batchsize: List[int] = single_input_shape[1:]  # type: ignore

            if self.data_format == "channels_last":
                channel_dim = single_input_shape_wo_batchsize[-1]
                if channel_dim % 2 != 0:
                    raise ValueError()
                target_shape = list(single_input_shape_wo_batchsize[:-2]) + [int(channel_dim / 2), 2]
            else:
                channel_dim = single_input_shape_wo_batchsize[0]
                if channel_dim % 2 != 0:
                    raise ValueError()
                target_shape = [2, int(channel_dim / 2)] + list(single_input_shape_wo_batchsize[1:])

            self.params_max = []
            self.params_min = []

            if self.finetune and self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
                self.beta_max = self.add_weight(
                    shape=target_shape, initializer="ones", name="beta_max", regularizer=None, constraint=ClipAlpha()
                )
                self.beta_min = self.add_weight(
                    shape=target_shape, initializer="ones", name="beta_max", regularizer=None, constraint=ClipAlpha()
                )
                self.params_max = [self.beta_max]
                self.params_min = [self.beta_min]

            self.op_reshape_in = DecomonReshape(tuple(target_shape), mode=self.mode)
            self.op_reshape_out = DecomonReshape(tuple(single_input_shape_wo_batchsize), mode=self.mode)

        def reset_layer(self, layer: Layer) -> None:
            pass
