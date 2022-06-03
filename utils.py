import os
import shutil
import json
import math

from collections import OrderedDict
import numpy as np

import jax
import jax.numpy as jnp
from jax import tree_util

from flax import jax_utils
from flax.training import checkpoints

def save_code_and_augments(args):
    """
        Save source code and arguments on train path to record what you did to get the results.

        Args:
            args: Arguments given in main code.

    """
    if os.path.isdir(args.train_path): 
        print ('='*50)
        print ('The folder already is. It will be overwrited')
        print ('='*50, '\n')
    else:
        os.mkdir(args.train_path)

    if not(os.path.isdir(os.path.join(args.train_path,'codes'))):
        shutil.copytree(
            args.home_path,
            os.path.join(args.train_path,'codes'),
            copy_function = shutil.copy, 
            ignore = shutil.ignore_patterns('*.pyc','__pycache__','*.swp', '.git')
        ) 

    with open(os.path.join(args.train_path, 'arguments.txt'), 'w') as f:
        json.dump(OrderedDict(args.__dict__), f, indent=4)

def shard(x, devices):
    """
        Shard data into each device.

        Args:
            x: data to be sharded.

        Return:
            sharded data

    """
    B, *D = x.shape
    num_devices = len(devices)
    return jnp.reshape(x, [num_devices, B//num_devices, *D])

def profile_model(arch, input_size, state, dtype, log = True):
    """
        Profile a flax model. FLOPS and number of parameters are calcluated acording to layer addon.

        Args:
            arch: name of model
            input_size: input data size
            state: Simple train state for the common case with a single Optax optimizer.
            dtype: model data type

        Return:
            flops: FLOPS of model
            n_params: number of parameters in model

    """

    cpu = jax.local_devices(backend = 'cpu')[0]

    input_size = (1, *input_size)
    variables = {'params': tree_util.tree_map(lambda x: jax.device_put(x, cpu), state.params), 'batch_stats': tree_util.tree_map(lambda x: jax.device_put(x, cpu), state.batch_stats)}
    variables = jax_utils.unreplicate(variables)

    _, state = state.apply_fn(variables, jnp.ones(input_size, dtype), train = False, mutable=['flops', 'n_params'])

    flops = int(sum(tree_util.tree_leaves(state['flops'])))
    n_params = int(sum(tree_util.tree_leaves(state['n_params'])))
    
    Prefix = ['','K','M','G']
    pf = (len(str(flops)) - 1)//3
    pp = (len(str(n_params)) - 1)//3

    if log:
        print('='*50)
        print('Model profile of ' + arch)
        print('- Model FLOPS   : {0:s} {1:s}'.format( str(round(flops/ 10**(pf*3), 2)).rjust(6), Prefix[pf] ))
        print('- Model #Params : {0:s} {1:s}'.format( str(round(n_params/ 10**(pp*3), 2)).rjust(6), Prefix[pp] ))
        print('='*50, '\n')

    return flops, n_params

class summary:
    """
        This summary recorder and it's objects works similar to accumulator in tensorflow.keras.metrics.

        Usage:
            >>> logger = utils.summary(args.do_log)
            >>> logger.assign(metrics, num_data = batch['image'].shape[1])
            >>> local_result = logger.result(metrics.keys())
            >>> print(local_result)
            >>> {'train': 90.00, 'loss': 0.1}
            >>> logger.reset(metrics.keys())

    """

    def __init__(self, ):
        self.holder = {}

    def assign(self, key_value, num_data = 1):
        for k, v in key_value.items():
            v = np.array(jax_utils.unreplicate(v)).item()
            if k in self.holder:
                self.holder[k] = [self.holder[k][0] + v * num_data, self.holder[k][1] + num_data]
            else:
                self.holder[k] = [v * num_data, num_data]

    def reset(self, keys = None):
        if keys is None:
            self.holder = {}
        else:
            for k in keys:
                del self.holder[k]

    def result(self, keys):
        return {k: self.holder[k][0]/self.holder[k][1] for k in keys}
