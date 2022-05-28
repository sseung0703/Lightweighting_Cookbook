from typing import Any

import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

class LayerAddon:
    '''
        Layer addon for various purposes such as
            1. profile the given layer (flops and number of parameter calculations)
            2. Compute the filter importance and mask (will be implemented)
            3. Store features for the feature distillation (will be implemented)
        When the layer is wrapped by LayerAddon, the usage of a layer is changed, so please check the below "Usage".

        This module works well but doesn't look fancy, so it will be improved more.

        Usage:
            >>> conv = LayerAddon(nn.Conv)
            >>> conv = partial(self.conv.profiled_call, use_bias=False, dtype=self.dtype)
            >>> x = conv(self.num_filters, (3, 3), name='conv_init', inputs = x)

    '''
    def __init__(self, layer):
        self.layer = layer

        def addon_call(*args, **kargs):
            x = kargs.pop('inputs')
            keep_feats = kargs.pop('keep_feats')
            
            layer_ = self.layer(*args, **kargs)
            y = layer_(x)

            for target_layer, feat_type in keep_feats:
                if target_layer == layer_.name:
                    if feat_type == 'in':
                        layer_.sow('keep_feats', 'keep_feats', [target_layer, x])

                    elif feat_type == 'out':
                        layer_.sow('keep_feats', 'keep_feats', [target_layer, y])

            flops, n_params = self.profiling(layer_, x, y)
            layer_.sow('flops', 'flops', flops)
            layer_.sow('n_params', 'n_params', n_params)
            return y

        self.addon_call = addon_call

    def profiling(self, layer_, x, y):
        ## Profiling for each layer type
        if self.layer == nn.Conv:
            Di = x.shape[-1]
            _, H, W, Do = y.shape

            kernel_size = layer_.kernel_size[0] * layer_.kernel_size[1] * Di * Do
            tensor_size = H*W

            flops = tensor_size * kernel_size
            n_params = kernel_size

            if layer_.use_bias:
                n_params += Do

        elif self.layer == nn.Dense:
            Di = x.shape[-1]
            Do = y.shape[-1]
            kernel_size = Di * Do

            tensor_size = 1
            for n in y.shape[1:-1]:
                tensor_size *= n

            flops = tensor_size * kernel_size
            n_params = kernel_size

            if layer_.use_bias:
                n_params += Do

        elif self.layer == nn.BatchNorm:
            Do = y.shape[-1]

            tensor_size = 1
            for n in y.shape[1:-1]:
                tensor_size *= n

            flops = tensor_size * Do * (1 + layer_.use_scale)
            n_params = Do * (2 + layer_.use_bias + layer_.use_scale)

        else:
            raise NotImplementedError(
                'Profile function of %s is not implemented\
                 If you want to use profile this, please implement yourself or report on Issue'%(self.layer)
            )
        return flops, n_params

