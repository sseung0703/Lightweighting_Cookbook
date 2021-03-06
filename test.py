import os
import time
import argparse
import warnings

import numpy as np

import jax
import jax.numpy as jnp

from flax.jax_utils import replicate
from flax.core import FrozenDict
from nets import ResNet
from dataloader import CIFAR
import op_utils
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='')

parser.add_argument("--arch", default='ResNet56', type=str,
                help = 'network architecture, currently only ResNet family is available')
parser.add_argument("--trained_param", type=str,
                help = 'trained parameter or checkpoint directory to be restored.\
                        If state of main model is restored, training will start at checkpoint.')
parser.add_argument("--data_path", type=str,
                help = 'Home directory of dataset for large datasets')
parser.add_argument("--dataset", default='CIFAR10', type=str,
                help = 'trained dataset, currently only CIFAR datasets are available')

parser.add_argument("--val_batch_size", default=250, type=int)

parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu_id", default= ['0'], type=str, nargs = '+',
                help = 'select which gpu will be used. Usage: --gpu_id 0 1 2')
args = parser.parse_args()

args.home_path = os.path.dirname(os.path.abspath(__file__))
os.environ["CUDA_VISIBLE_DEVICES"]=",".join(args.gpu_id)

print(f"\n Detected device: {jax.local_devices()}\n")

if __name__ == '__main__':
    rng = jax.random.PRNGKey(args.seed)
    rng, key = jax.random.split(rng)
    datasets = CIFAR.build_dataset_providers(args, key, test_only = True)

    model_cls = getattr(ResNet, args.arch)
    model = model_cls(num_classes = datasets.num_classes)
    
    model, state = op_utils.restore_checkpoint(model, args.trained_param)
    
    features_dict = {}
    def pruning(keys, variables):
        v = variables[list(variables.keys())[0]]
        features_dict[keys[-1]] = v.shape[-1]

    def rebuild_tree(frozen_dict, K):
        new_frozen_dict = {}
        for k, layer in frozen_dict.items():
            if 'mask' not in k:
                if any([ not(isinstance(p, FrozenDict) or isinstance(p, dict)) for _, p in layer.items()]) and layer:
                    pruning(K+[k], layer)
                else:
                    rebuild_tree(layer, K+[k])
    rebuild_tree(state['params'], [])
    model.features_dict = features_dict

    eval_state = op_utils.EvalState.create(apply_fn = model.apply, params = state['params'], batch_stats = state['batch_stats'])
    eval_step = op_utils.create_eval_step(datasets.num_classes)

    utils.profile_model(args.arch, datasets.input_size, eval_state, model.dtype)

    logger = utils.summary()
    for batch in datasets.provider['test']():
        metrics = eval_step(eval_state, batch)
        metrics = {'test/' + k: v for k, v in metrics.items()}
        
        logger.assign(metrics, num_data = batch['image'].shape[1])

    eval_result = logger.result(metrics.keys())
    print ('Test loss = {0:0.4f}, Test acc = {1:0.2f}'.format(eval_result['test/loss'], eval_result['test/accuracy']))

