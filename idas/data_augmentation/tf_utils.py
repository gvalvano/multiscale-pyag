import tensorflow as tf


def flip_tensors_left_right(list_of_tensors, probability=0.5):
    """Randomly flip a list of tensors (i.e. image and mask).
    """
    assert 0.0 <= probability <= 1.0

    uniform_random = tf.random_uniform([], 0, 1.0)
    flip_cond = tf.less(uniform_random, probability)
    augmented = []
    for tensor in list_of_tensors:
        augmented.append(tf.cond(flip_cond, lambda: tf.image.flip_left_right(tensor), lambda: tensor))

    return augmented


def flip_tensors_up_down(list_of_tensors, probability=0.5):
    """Randomly flip a list of tensors (i.e. image and mask).
    """
    assert 0.0 <= probability <= 1.0

    uniform_random = tf.random_uniform([], 0, 1.0)
    flip_cond = tf.less(uniform_random, probability)
    augmented = []
    for tensor in list_of_tensors:
        augmented.append(tf.cond(flip_cond, lambda: tf.image.flip_up_down(tensor), lambda: tensor))

    return augmented
