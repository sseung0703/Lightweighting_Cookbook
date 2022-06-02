# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flax implementation of ResNet V1."""

from functools import partial
from typing import Any, Callable, Sequence, Tuple, Dict
from os.path import split

from flax import linen as nn
import jax.numpy as jnp

from nets.net_utils import LayerAddon

ModuleDef = Any

class ResNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    name = ''

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(features = self.filters, kernel_size = (3, 3), strides = self.strides, name = self.name + '/conv0', inputs = x)
        y = self.norm(name = self.name + '/bn0', inputs = y)
        y = self.act(y)

        y = self.conv(features = self.filters, kernel_size = (3, 3), name = self.name + '/conv1', inputs = y)
        y = self.norm(scale_init=nn.initializers.zeros, name = self.name + '/bn1', inputs = y)

        if residual.shape != y.shape:
            residual = self.conv(features = self.filters, kernel_size = (1, 1), strides = self.strides, name = self.name + '/conv_proj', inputs = residual)
            residual = self.norm(name= self.name + '/bn_proj', inputs = residual)

        return self.act(residual + y)

class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(features = self.filters, kernel_size = (1, 1), name = self.name + '/conv0', inputs = x)
        y = self.norm(name = self.name + '/bn0', inputs = y)
        y = self.act(y)

        y = self.conv(features = self.filters, kernel_size = (3, 3), strides = self.strides, name = self.name + '/conv1', inputs = y)
        y = self.norm(name = self.name + '/bn1', inputs = y)
        y = self.act(y)

        y = self.conv(features = self.filters * 4, kernel_size = (1, 1), name = self.name + '/conv2', inputs = x)
        y = self.norm(scale_init=nn.initializers.zeros, name = self.name + '/bn2', inputs = y)

        if residual.shape != y.shape:
            residual = self.conv(features = self.filters * 4, kernel_size = (1, 1), strides = self.strides, name = self.name + '/conv_proj', inputs = residual)
            residual = self.norm(name= self.name + '/bn_proj', inputs = residual)

        return self.act(residual + y)

class ResNet(nn.Module):
    """ResNetV1."""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    keep_feats: Sequence[str]
    features_dict: Sequence[str]
    mask_dict: Dict[str, ModuleDef]
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    conv: ModuleDef = LayerAddon(nn.Conv)
    norm: ModuleDef = LayerAddon(nn.BatchNorm)

    @nn.compact
    def __call__(self, x, train: bool = True):
        keep_feats = [split(kp) for kp in self.keep_feats]

        conv = partial(self.conv.addon_call, use_bias=False, dtype=self.dtype,
                keep_feats = keep_feats, mask_dict = self.mask_dict, features_dict = self.features_dict)
        norm = partial(self.norm.addon_call,
                       use_running_average=not train,
                       momentum=0.9,
                       epsilon=1e-5,
                       dtype=self.dtype,
                       keep_feats = keep_feats,
                       mask_dict = self.mask_dict)
        dense = partial(LayerAddon(nn.Dense).addon_call, keep_feats = keep_feats, mask_dict = self.mask_dict)
        
        if len(self.stage_sizes) == 3:
            x = conv(features = self.num_filters, kernel_size = (3, 3), name='conv_init', inputs = x)
            x = norm(name='bn_init', inputs = x)
            x = nn.relu(x)

        else:    
            x = conv(self.num_filters, (7, 7), (2, 2), name='conv_init', inputs = x)
            x = norm(name='bn_init', inputs = x)
            x = nn.relu(x)
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=strides,
                                   conv=conv,
                                   norm=norm,
                                   act=self.act,
                                   name = 'block_%d_%d'%(i,j),
                                   )(x)

            if ('stage','last') in keep_feats:
                self.sow('keep_feats', 'keep_feats', x)

        x = jnp.mean(x, axis=(1, 2))
        x = dense(features = self.num_classes, dtype=self.dtype, name = 'classifier', inputs = x)

        return x


## ImageNet confs
ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3],
                    block_cls=BottleneckResNetBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3],
                    block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3],
                    block_cls=BottleneckResNetBlock)

ResNet18Local = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                        block_cls=ResNetBlock, conv=nn.ConvLocal)

## CIFAR confs
ResNet56 = partial(ResNet, stage_sizes=[9, 9, 9],
                   block_cls=ResNetBlock,
                   num_filters = 16,
                   keep_feats = [],
                   mask_dict = {},
                   features_dict = {}
                   )

ResNet32 = partial(ResNet, stage_sizes=[5, 5, 5],
                   block_cls=ResNetBlock,
                   num_filters = 16,
                   keep_feats = [],
                   mask_dict = {},
                   features_dict = {}
                   )

ResNet20 = partial(ResNet, stage_sizes=[3, 3, 3],
                   block_cls=ResNetBlock,
                   num_filters = 16,
                   keep_feats = [],
                   mask_dict = {},
                   features_dict = {}
                   )


# Used for testing only.
_ResNet1 = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
_ResNet1Local = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock,
                        conv=nn.ConvLocal)
