import os
from typing import Any, Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import tree_util

from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from flax import core
from flax import struct
import optax

import op_utils

def create_distill_step(weight_decay, distill_objective):
    """
        create training step with given knowledge distillation objective function
        This function is based on the below flax example.
        https://github.com/google/flax/blob/main/examples/imagenet/train.py

        Args:
            weight_decay: L2 regularization strength
            distill_objective: Objective function to train student network with teacher knowledge. you can find each detail of objective function in distiller/*

        return:
            distill_step: training step for given model and trainable variables with knowledge distillation. 
                whole computations are jit compiled and pmapped 
            sync_batch_stats: synchronize each batch stats on multiple devices.

    """
    @jax.jit
    def distill_step(state, teacher_state, batch):
        def forward(params):
            variables = {'params': params, 'batch_stats': state.batch_stats}
            logits, new_state = state.apply_fn(variables, batch['image'], mutable=['batch_stats', 'keep_feats'])

            teacher_variables = {'params': teacher_state.params, 'batch_stats': teacher_state.batch_stats}
            teacher_logits, new_teacher_state = teacher_state.apply_fn(teacher_variables, batch['image'], train = False, mutable = ['keep_feats'])

            # objective function
            loss = distill_objective(logits, new_state['keep_feats'], new_teacher_state['keep_feats'], batch['label'])
            return loss, (new_state, logits, loss)

        grad_fn = jax.value_and_grad(forward, has_aux=True)
        aux, grads = grad_fn(state.params)
        new_state, logits, loss = aux[1]
        
        grads = jax.lax.pmean(grads, axis_name='batch')
        grads = op_utils.l2_weight_decay(state.params, grads, weight_decay)

        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
        new_state = state.apply_gradients(
            grads=grads, batch_stats=new_state['batch_stats'])

        metrics = {
            'loss': loss,
            'accuracy': accuracy * 100,
        }
        metrics = jax.lax.pmean(metrics, axis_name='batch')

        return new_state, teacher_state, metrics

    distill_step = jax.pmap(distill_step, axis_name = "batch")

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    def sync_batch_stats(state):
        return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

    return distill_step, sync_batch_stats


