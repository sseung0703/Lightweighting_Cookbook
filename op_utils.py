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


def initialized(key, input_size, model):
    """
        initialize given model's parameters using PRNG.

        Args:
            args: Arguments given in main code.
            rng: a PRNG key used as the random key.
            test_only: If this is set to True, only test data provider will be generated.

        Return:
            variables['params']: initialized parmas
            variables['batch_stats']: initialized batch statis params such as moving mean and variance of batch normalization
    """

    input_size = (1, *input_size)
    
    @jax.jit
    def init(*args):
        return model.init(*args)
    variables = init({'params': key}, jnp.ones(input_size, model.dtype))
    return variables['params'], variables['batch_stats']

class TrainState(train_state.TrainState):
    batch_stats: Any

class EvalState(struct.PyTreeNode):
    """
        Build a PyTreeNode for the inference only state, e.g., evaluation, teacher knowledge extraction.
        Compare to TrainState it doesn't require tx (optimizer), learning_rate_fn.

        Args:
            apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
                convenience to have a shorter params list for the `eval_step()` function
                in your evaluation loop.
            params: The parameters to be used by `apply_fn`.
    """

    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    batch_stats: Any

    @classmethod
    def create(self, *, apply_fn, params, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        return self(
            apply_fn=apply_fn,
            params=params,
            **kwargs,
        )

def l2_weight_decay(params, grads, weight_decay):
    """
        Apply l2 regularization to trainable parameters.
        Note that appling regularization to gradients rather than variables is more efficient.

        Args:
            params: Paramteters to be trained and regularized.
            grads: Gradients for trained parameters.
            weight_decay: regularization strength.

        return:
            new_grads: Gradients for trained parameters with regularization constraints.
    """

    params_flat, treedef = jax.tree_flatten(params)
    grads_flat = treedef.flatten_up_to(grads)
    grads_flat = [grad + param * weight_decay for param, grad in zip(params_flat, grads_flat) ]
    new_grads = jax.tree_unflatten(treedef, grads_flat)
    return new_grads

def create_train_state_n_step(args, rng, model, input_size, num_classes, learning_rate_fn):
    """
        Initialize components for training and create training step.
        This function is based on the below flax example.
        https://github.com/google/flax/blob/main/examples/imagenet/train.py

        Args:
            args: General arguments.
            rng: a PRNG key used as the random key.
            model: FLAX model 
            num_classes: number of classes
            learning_rate_fn: leanrning rate scheduler built by optax.

        return:
            train_step: training step for given model and trainable variables. whole computations are jit compiled and 
            sync_batch_stats: synchronize each batch stats on multiple devices.
            state: Simple train state for the common case with a single Optax optimizer.
                details can be found at https://github.com/google/flax/blob/main/flax/training/train_state.py

    """

    params, batch_stats = initialized(rng, input_size, model)
    variables = {'params': params, 'batch_stats': batch_stats}

    tx = optax.sgd(
        learning_rate = learning_rate_fn,
        momentum=0.9,
        nesterov=True,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )
    state = jax_utils.replicate(state)

    @jax.jit
    def train_step(state, batch):
        def forward(params):
            variables = {'params': params, 'batch_stats': state.batch_stats}
            logits, new_model_state = state.apply_fn(variables, batch['image'], mutable=['batch_stats'])

            # objective function
            one_hot_labels = common_utils.onehot(batch['label'], num_classes=num_classes)
            loss = jnp.mean( optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels) )
            return loss, (new_model_state, logits, loss)

        grad_fn = jax.value_and_grad(forward, has_aux=True)
        aux, grads = grad_fn(state.params)
        new_model_state, logits, loss = aux[1]
        
        grads = jax.lax.pmean(grads, axis_name='batch')
        grads = l2_weight_decay(state.params, grads, args.weight_decay)

        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
        new_state = state.apply_gradients(
            grads=grads, batch_stats=new_model_state['batch_stats'])

        metrics = {
            'loss': loss,
            'accuracy': accuracy * 100,
        }
        metrics = jax.lax.pmean(metrics, axis_name='batch')

        return new_state, metrics

    train_step = jax.pmap(train_step, axis_name = "batch")

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    def sync_batch_stats(state):
        return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

    return train_step, sync_batch_stats, state

def create_eval_step(num_classes):
    """
        This function is based on the below flax example.
        https://github.com/google/flax/blob/main/examples/imagenet/train.py

        Args:
            model: FLAX model 
            num_classes: number of classes

        return:
            eval_step: evaluation step

    """
    @jax.jit
    def eval_step(state, batch):
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)

        # objective function
        one_hot_labels = common_utils.onehot(batch['label'], num_classes=num_classes)
        loss = jnp.mean( optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels) )

        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
        metrics = {
            'loss': loss,
            'accuracy': accuracy * 100,
        }
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        return metrics

    eval_step = jax.pmap(eval_step, axis_name = "batch")
    return eval_step

def save_checkpoint(state, train_path, epoch):
    """
        Save checkpoint of flax model.

        Args:
            state: model parameters to be saved.
            train_path: str or pathlib-like path to store checkpoint files in.
            epoch: subname of checkpoint

    """

    if jax.process_index() == 0:
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(train_path, state, epoch, keep=3)

def restore_checkpoint(trained_param, state = None):
    """
        Restore checkpoint of flax model.
        If state is given, components of state will be replaced by checkpoints,
        otherwise, dictionary type of checkpoints is returned.

        Args:
            state: model parameters to be saved.
            train_path: str or pathlib-like path to store checkpoint files in.
            epoch: subname of checkpoint

        return:
            restored_state: PyTreeNode or dictionary type of checkpoint
    """

    if os.path.isfile(trained_param):
        ckpt_dir, prefix = os.path.split(trained_param)
    else:
        ckpt_dir = trained_param
        prefix = 'checkpoint_'

    restored_state = checkpoints.restore_checkpoint(ckpt_dir, state, prefix = prefix)
    restored_state = jax_utils.replicate(restored_state)
    return restored_state


