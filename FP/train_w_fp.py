import os
import sys
import time
import argparse
import warnings
import importlib

import numpy as np

import jax
import jax.numpy as jnp
from jax import tree_util

from flax.metrics import tensorboard
import optax

sys.path.append(os.path.split(os.getcwd())[0])

from nets import ResNet
from dataloader import CIFAR
import op_utils
import utils

import prune_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='')

## Student arguments
parser.add_argument("--train_path", default="../test", type=str,
        help = 'training path to save results filing including source code, checkpoint, and tensorboard log')
parser.add_argument("--arch", default='ResNet32', type=str,
        help = 'network architecture, currently only ResNet family is available')
parser.add_argument("--trained_param", type=str,
        help = 'trained parameter or checkpoint directory to be restored.\
                If state of main model is restored, training will start at checkpoint.')
parser.add_argument("--data_path", type=str,
        help = 'Home directory of dataset for large datasets')
parser.add_argument("--dataset", default='CIFAR10', type=str,
        help = 'trained dataset, currently only CIFAR datasets are available')

## Filter pruning arguments
parser.add_argument("--criterion", default = 'FilterNorm', type=str,
        help = 'criterion to score filter importance.')

parser.add_argument("--strategy", default = 'AtOnce', type=str,
        help = 'strategy to prune the filters')

parser.add_argument("--frr", default = 0.5, type=float)

parser.add_argument("--learning_rate", default = 1e-1, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--decay_points", default = [.3, .6, .8], type=float, nargs = '+')
parser.add_argument("--decay_rate", default=.2, type=float)

parser.add_argument("--train_epoch", default=200, type=int)
parser.add_argument("--batch_size", default = 128, type=int)
parser.add_argument("--val_batch_size", default=250, type=int)

parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu_id", default= ['0'], type=str, nargs = '+',
        help = 'select which gpu will be used. Usage: --gpu_id 0 1 2')
parser.add_argument("--do_log", default=200, type=int,
        help = 'logging period')
parser.add_argument("--deterministic", default = False, action = 'store_true',
        help = 'For the purpose of reproducing, deterministic flag should be enabled to ensure the algorithm used for dense computation such as convolution.\
                Even you properly set the flags or PRNG, the result may be different from this tutorial because the algorithm tuning and/or the hardware unit varies depending on the GPU generation.')

args = parser.parse_args()
args.home_path = os.getcwd()
os.environ['CUDA_VISIBLE_DEVICES']=','.join(args.gpu_id)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = '0'
if args.deterministic:
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

print(f"\n Detected device: {jax.local_devices()}\n")

if __name__ == '__main__':
    rng = jax.random.PRNGKey(args.seed)
    utils.save_code_and_augments(args)

    rng, key = jax.random.split(rng)
    datasets = CIFAR.build_dataset_providers(args, key)

    ## Model building
    if 'ResNet' in args.arch:
        model_cls = getattr(ResNet, args.arch)
        model = model_cls(num_classes = datasets.num_classes)

    model.mask_dict = prune_utils.MaskInitialization(args.arch, model)
    criterion = importlib.import_module('criteria.' + args.criterion)
    model.mask_dict['criterion'] = criterion.measure

    learning_rate_fn = optax.piecewise_constant_schedule(args.learning_rate, { int(dp * args.train_epoch * datasets.iter_len['train']) : args.decay_rate for dp in args.decay_points})
    rng, key = jax.random.split(rng)
    state = op_utils.create_train_state(key, model, datasets.input_size, learning_rate_fn)

    ori_flops, ori_n_params = utils.profile_model(args.arch, datasets.input_size, state, model.dtype)

    if args.trained_param is not None:
        old_state = op_utils.restore_checkpoint(args.trained_param)
        state = state.replace( params = state.params.copy(old_state['params']), batch_stats = old_state['batch_stats'] ) 
    start_epoch = 0

    if args.strategy == 'AtOnce':
        strategy = importlib.import_module('strategy.' + args.strategy)
        state = strategy.prune(state, datasets, args.frr, ori_flops)

    else:
        raise NotImplementedError(
            'Only offline transfer strategy is available currently.'
        )
    
    model, params, batch_stats = prune_utils.actual_pruning(args.arch, model, datasets.input_size, state)
    rng, key = jax.random.split(rng)
    state = op_utils.create_train_state(key, model, datasets.input_size, learning_rate_fn, params, batch_stats)

    train_step, sync_batch_stats = op_utils.create_train_step(args.weight_decay)
    eval_step = op_utils.create_eval_step(datasets.num_classes)

    tic = time.time()
    logger = utils.summary()

    summary_writer = tensorboard.SummaryWriter(args.train_path)

    for epoch in range(start_epoch, args.train_epoch):
        # Train loop
        for batch in datasets.provider['train']():
            state, metrics = train_step(state, batch)
            metrics = {'train/' + k: v for k, v in metrics.items()}
            
            logger.assign(metrics, num_data = batch['image'].shape[1])
            
            if state.step % args.do_log == 0:
                train_time = time.time() - tic
                
                local_result = logger.result(metrics.keys())
                print ('Global step {0:6d}: loss = {1:0.4f}, acc = {2:0.2f} ({3:1.3f} sec/step)'.format(state.step.item(), local_result['train/loss'], local_result['train/accuracy'], train_time/args.do_log))    
                
                tic = time.time()
        epoch += 1
        state = sync_batch_stats(state)
        op_utils.save_checkpoint(state, args.train_path, epoch)

        train_result = logger.result(metrics.keys())
        for k, v in train_result.items():
            summary_writer.scalar(k, v, epoch)
 
        test_tic = time.time()

        # Test loop    
        for batch in datasets.provider['test']():
            metrics = eval_step(state, batch)
            metrics = {'test/' + k: v for k, v in metrics.items()}
        
            logger.assign(metrics, num_data = batch['image'].shape[1])

        eval_result = logger.result(metrics.keys())
        print('='*50)
        print ('Epoch {0:3d}:\n\tTest loss = {1:0.4f}, Test acc = {2:0.2f}'.format(epoch, eval_result['test/loss'], eval_result['test/accuracy']))
        print('='*50)

        for k, v in eval_result.items():
            summary_writer.scalar(k, v, epoch)

        summary_writer.flush()
        logger.reset()

        # Time compensation
        tic = tic + time.time() - test_tic

