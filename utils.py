import os
import shutil
import json
import math

from collections import OrderedDict
import numpy as np

import jax
import jax.numpy as jnp

from flax import jax_utils
from flax.training import checkpoints

def save_code_and_augments(args):
    """
        Save source code and arguments on train path to record what you did to get the results.

        Args:
            args: Arguments given in main code.

    """
    if os.path.isdir(args.train_path): 
        print ('============================================')
        print ('The folder already is. It will be overwrited')
        print ('============================================')
    else:
        os.mkdir(args.train_path)

    if not(os.path.isdir(os.path.join(args.train_path,'codes'))):
        shutil.copytree(
            args.home_path,
            os.path.join(args.train_path,'codes'),
            copy_function = shutil.copy, 
            ignore = shutil.ignore_patterns('*.pyc','__pycache__','*.swp')
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

