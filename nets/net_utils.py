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
            >>> x = conv(self.num_filters, (3, 3), name='conv_init', inputs = x, keep_feats = keep_feats, mask_dict = self.mask_dict)

    '''
    def __init__(self, layer):
        self.layer = layer

        def addon_call(*args, **kargs):
            x = kargs.pop('inputs')
            keep_feats = kargs.pop('keep_feats')
            mask_dict = kargs.pop('mask_dict')

            if 'features_dict' in kargs:
                features_dict = kargs.pop('features_dict')
                kargs['features'] = features_dict.get(kargs['name'], kargs['features'])

            layer_ = self.layer(*args, **kargs)

            ## Forwad with mask for filter pruning
            y, in_mask, out_mask = self.masked_forward(layer_, mask_dict, x)

            ## Keep asked features for various perposes.
            self.keep_asked_feat(layer_, keep_feats)

            ## Profile this layer considering mask
            flops, n_params = self.profiling(layer_, x, y, in_mask, out_mask)
            layer_.sow('flops', 'flops', flops)
            layer_.sow('n_params', 'n_params', n_params)
            return y

        self.addon_call = addon_call

    def profiling(self, layer_, x, y, in_mask, out_mask):
        Di = x.shape[-1] if in_mask is None else jnp.sum(in_mask).astype(jnp.int32)
        Do = y.shape[-1] if out_mask is None else jnp.sum(out_mask).astype(jnp.int32)

        if self.layer == nn.Conv:
            _, H, W, _ = y.shape

            kernel_size = layer_.kernel_size[0] * layer_.kernel_size[1] * Di * Do
            tensor_size = H*W

            flops = tensor_size * kernel_size
            n_params = kernel_size

            if layer_.use_bias:
                n_params += Do

        elif self.layer == nn.Dense:
            kernel_size = Di * Do

            tensor_size = 1
            for n in y.shape[1:-1]:
                tensor_size *= n

            flops = tensor_size * kernel_size
            n_params = kernel_size

            if layer_.use_bias:
                n_params += Do

        elif self.layer == nn.BatchNorm:
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

    def masked_forward(self, layer_, mask_dict, x):
        name = layer_.name

        if name + '/in_mask' in mask_dict:
            in_mask = mask_dict[name + '/in_mask'](x)
            ## In most cases, input feature map doesn't requires masking.
        else:
            in_mask = None

        y = layer_(x)

        if name + '/out_mask' in mask_dict:
            out_mask = mask_dict[name + '/out_mask'](x)

            if name.replace('conv','bn') +'/out_mask' in mask_dict:
                ## If BachNorm follows this layer, mask should be applies after normalization. 
                y = y * out_mask

            ## Importance is only gathered in the output mask to avoid duplication.
            mask_dict[name + '/out_mask'].sow('importance', 'importance', mask_dict['criterion'](name, layer_, x, y))

        else:
            out_mask = None

        layer_.sow('in_mask', 'in_mask', in_mask, reduce_fn = lambda xs, x: x, init_fn = lambda :None)
        layer_.sow('out_mask', 'out_mask', out_mask, reduce_fn = lambda xs, x: x, init_fn = lambda :None)
        return y, in_mask, out_mask

    def keep_asked_feat(self, layer_, keep_feats):
        for target_layer, feat_type in keep_feats:
            if target_layer == layer_.name:
                if feat_type == 'in':
                    layer_.sow('keep_feats', 'keep_feats', x)

                elif feat_type == 'out':
                    layer_.sow('keep_feats', 'keep_feats', y)

