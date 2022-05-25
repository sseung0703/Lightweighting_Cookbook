import jax
import jax.numpy as jnp

# This code is highly inspired by the below awesome tutorials.
# If you want to know JAX more, please there.
# https://www.kaggle.com/code/aakashnain/building-models-in-jax-part2-flax/notebook

def identity(img):
    """Returns an image as it is."""
    return img

def random_horizontal_flip(rng, img):
    """Randomly flip an image horizontally.
    
    Args:
        img: Array representing the image
        rng: Pseudo Random Numper Generator (PRNG)
    Returns:
        Flipped or an identity image
    """
    cond = jax.random.bernoulli(rng)
    return jax.lax.cond(cond, jnp.fliplr, identity, img)

def random_crop_pad(rng, img, pad, mode = 'constant'):
    """Randomly crop an image with padding.
    
    Args:
        img: Array representing the image
        rng: Pseudo Random Numper Generator (PRNG)
        pad: interger, pad size before cropping
        mode: string, padding mode, you can find available modes in "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html?highlight=pad"
    Returns:
        Random cropeed or an identity image
    """
    
    rng, key = jax.random.split(rng) 
    cond = jax.random.bernoulli(key)

    rng, key = jax.random.split(rng)
    h_init, w_init = jax.random.randint(key, [2], 0, pad*2)

    def aug(img):
        shape = img.shape
        img = jnp.pad(img, [[pad, pad], [pad, pad], [0,0]], mode = mode)
        img = jax.lax.dynamic_slice(img, (h_init, w_init,0), shape)
        return img

    return jax.lax.cond(cond, aug, identity, img)
 
