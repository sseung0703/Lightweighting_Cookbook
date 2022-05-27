import os
from functools import partial
import numpy as np

import jax
import jax.numpy as jnp

from flax.jax_utils import prefetch_to_device

from dataloader import Augmentation as aug
from utils import shard

# This code is highly inspired by the below awesome tutorials.
# If you want to know JAX more, please there.
# https://www.kaggle.com/code/aakashnain/building-models-in-jax-part2-flax/notebook


class build_dataset_providers:
    """
        Build CIFAR dataset providers according to given arguments and devices.
        including train and test

        Args:
            args: Arguments given in main code.
            rng: a PRNG key used as the random key.
            test_only: If this is set to True, only test data provider will be generated.

        Usage:
            >>> datasets = CIFAR.build_dataset_providers(args, key)
            >>> for data in datasets.provider['train']():
    """
    def __init__(self, args, rng, test_only = False):
        self.args = args

        if args.dataset == 'CIFAR10':
            train_images, train_labels, test_images, test_labels = self.Cifar10()
        if args.dataset == 'CIFAR100':
            train_images, train_labels, test_images, test_labels = self.Cifar100()

        self.num_classes = int(args.dataset[5:])
        self.input_size = [32,32,3]

        self.devices = jax.local_devices()
        
        self.cardinality = {}
        self.iter_len = {}
        self.rng = {}

        self.provider = {}

        if not(test_only):
            self.rng['train'] = rng
            self.provider['train'] = self.gen_provider('train',
                                                       {'image': train_images, 'label': train_labels},
                                                       args.batch_size,
                                                       shuffle = True,
                                                       rng = self.rng['train'],
                                                       drop_remainder = True)
        self.rng['test'] = rng
        self.provider['test'] = self.gen_provider('test', 
                                                  {'image': test_images, 'label': test_labels},
                                                  args.val_batch_size,
                                                  rng = self.rng['test'],
                                                  drop_remainder = False)
 
        print('='*50)
        print(args.dataset, 'data providers are built as follows:')
        print('- Data cardinality     :',self.cardinality)
        print('- Number of iterations :',self.iter_len)
        print('='*50, '\n')

    def gen_provider(self, 
                     split,
                     data,
                     batch_size,
                     shuffle = False,
                     drop_remainder = True,
                     rng = None):

        self.cardinality[split] = data['image'].shape[0]
        self.iter_len[split] = data['image'].shape[0] // batch_size if drop_remainder \
                          else int(np.ceil(data['image'].shape[0] / batch_size))

        @partial(jax.vmap, in_axes=(0, 0))
        def pre_processing(rng, img):
            rng, key = jax.random.split(rng)
            img = aug.random_horizontal_flip(key, img)

            rng, key = jax.random.split(rng)
            img = aug.random_crop_pad(key, img, 4, mode = 'reflect')

            ## add additional augmentation scheme here
            return img

        pre_processing = jax.jit(pre_processing)

        def provider():
            indices = np.arange(self.cardinality[split])

            if shuffle:
                self.rng[split], key = jax.random.split(self.rng[split])
                indices = jax.random.shuffle(key, indices)

            for batch in range(self.iter_len[split]):
                curr_idx = indices[batch * batch_size: (batch+1) * batch_size]
                batch_data = {k: d[curr_idx] for k, d in data.items()}

                if split == 'train':
                    self.rng[split], key = jax.random.split(self.rng[split])
                    v_cond = jax.random.split(key, batch_size)

                    batch_data['image'] = jax.device_put(batch_data['image'], jax.devices("cpu")[0])
                    batch_data['image'] = pre_processing(v_cond, batch_data['image'])
                yield {k: shard(d, self.devices) for k, d in batch_data.items()}

        return lambda: prefetch_to_device(provider(), size = 10)
        
    def Cifar10(self,):
        from tensorflow.keras.datasets.cifar10 import load_data

        (train_images, train_labels), (val_images, val_labels) = load_data()
        train_images = (train_images-np.array([113.9,123.0,125.3]))/np.array([66.7,62.1,63.0])
        val_images = (val_images-np.array([113.9,123.0,125.3]))/np.array([66.7,62.1,63.0])
        return train_images, train_labels.reshape(-1), val_images, jnp.array(val_labels).reshape(-1)
    
    def Cifar100(self,):
        from tensorflow.keras.datasets.cifar100 import load_data

        (train_images, train_labels), (val_images, val_labels) = load_data()
        train_images = (train_images-np.array([112,124,129]))/np.array([70,65,68])
        val_images = (val_images-np.array([112,124,129]))/np.array([70,65,68])
        return train_images, train_labels.reshape(-1), val_images, jnp.array(val_labels).reshape(-1)
