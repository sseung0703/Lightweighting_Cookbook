import os

import jax
import jax.numpy as jnp
from jax import tree_util

from flax import jax_utils
import utils

def prune(state, datasets, frr, ori_flops, collect_importance):
    """
        Collect filter importance only one time and prune the network at once.
        There is only one pruning iteration, so pruning cost is much lower than others.
        This strategy looks too naive, but lots of algorithms, e.g., FPGM, Hrank, have adopted this strategy.

        Args:
            state:
            datasets:
            frr:
            ori_flops:
            collect_importance:

        return:
            state: state that contains pruned model, params, and batch_stats.
                   Note that, at this point, pruning is simulated by masking and "actual pruning" should be done.

    """
    ## Gather importance of each mask
    importance = collect_importance(state, datasets)
    
    th_list = jnp.sort(jnp.concatenate(tree_util.tree_leaves(importance), 0))
    mask_params = {k:p for k,p in state.params.items() if 'mask' in k}


    def imp2mask(p, imp, th):
        mask = jnp.logical_or(imp > th, imp == jnp.max(imp)).astype(jnp.float32)
        mask = jnp.stack([mask] * p['mask'].shape[0])
        p = p.copy({'mask': mask})
        return p


    def binarySearch(state, mask_params, th_list, l, r, target_frr):
        mid = l + (r - l) // 2
        th = th_list[mid]
        new_mask_params = {k: imp2mask(p, imp, th) for (k,p), (k2, imp) in zip(mask_params.items(), importance.items())}
        state = state.replace(params = state.params.copy(new_mask_params))
        cur_flops = utils.profile_model('', datasets.input_size, state, datasets.dtype, log = False)[0]
        
        cur_frr = 1 - cur_flops/ori_flops
        if cur_frr == target_frr or mid in [l,r]:
            if cur_frr < target_frr:
                th = th_list[mid+1]
                new_mask_params = {k: imp2mask(p, imp, th) for (k,p), (k2, imp) in zip(mask_params.items(), importance.items())}
                state = state.replace(params = state.params.copy(new_mask_params))

                cur_flops = utils.profile_model('', datasets.input_size, state, datasets.dtype, log = True)[0]
                print('{0:.2f}% of FLOPS pruned network'.format((1 -cur_flops/ori_flops)*100))
 
            else:
                cur_flops = utils.profile_model('', datasets.input_size, state, datasets.dtype, log = True)[0]
                print('{0:.2f}% of FLOPS pruned network'.format((1 -cur_flops/ori_flops)*100))

            return state
                                                      
        elif cur_frr > target_frr:
            return binarySearch(state, new_mask_params, th_list, l, mid-1, target_frr)
                                                                                          
        else:
            return binarySearch(state, new_mask_params, th_list, mid + 1, r, target_frr)
    
    state = binarySearch(state, mask_params, th_list, 0, len(th_list), frr)
    return state
