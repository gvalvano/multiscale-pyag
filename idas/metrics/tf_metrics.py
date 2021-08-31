#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tensorflow as tf


def dice_coe(output, target, axis=(1, 2, 3), smooth=1e-12):
    """Soft Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.

    Examples
    ---------
    '>>> outputs = tl.act.pixel_wise_softmax(network.outputs)'
    '>>> dice_loss = 1 - dice_coe(outputs, y_)'

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """

    assert output.dtype in [tf.float32, tf.float64]
    assert target.dtype in [tf.float32, tf.float64]

    intersection = tf.reduce_sum(output * target, axis=axis)

    a = tf.reduce_sum(output, axis=axis)
    b = tf.reduce_sum(target, axis=axis)

    score = (2. * intersection + smooth) / (a + b + smooth)
    score = tf.reduce_mean(score, name='dice_coe')
    return score


def generalized_dice_coe(output, target, axis=(1, 2, 3), smooth=1e-12):
    """Generalized Soft Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.
    This generalization of the Dice score weights each class over the number of pixels inside.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]`` for compatibility with dice_coe(). The actual reduction is on
        all dimensions but the last one.
    smooth : float
        This small value will be added to the numerator and denominator.

    Examples
    ---------
    '>>> outputs = tl.act.pixel_wise_softmax(network.outputs)'
    '>>> dice_loss = 1 - generalized_dice_coe(outputs, y_)'

    """

    assert output.dtype in [tf.float32, tf.float64]
    assert target.dtype in [tf.float32, tf.float64]

    intersection = tf.reduce_sum(output * target, axis=axis[:-1])
    a = tf.reduce_sum(output, axis=axis[:-1])
    b = tf.reduce_sum(target, axis=axis[:-1])

    # define weights for each class
    weights = 1.0 / ((tf.reduce_sum(target, axis=axis[:-1]) ** 2) + smooth)

    # numerator and denominator:
    numerator = tf.reduce_sum(weights * intersection, axis=1)
    denominator = tf.reduce_sum(weights * (a + b), axis=1)

    score = 2. * (numerator + smooth) / (denominator + smooth)
    score = tf.reduce_mean(score, name='dice_coe')
    return score


def shannon_binary_entropy(incoming, axis=(1, 2), unscaled=False, smooth=1e-12):
    """
    Evaluates shannon entropy on a binary mask (data type float). The last index contains one-hot encoded predictions.
    :param incoming: incoming tensor (one-hot encoded). On the first dimension there is the number of samples (typically
                the batch size)
    :param axis: axis containing the input dimension. Assuming 'incoming' to be a 4D tensor, axis has length 2: width
                and height; if 'incoming' is a 5D tensor, axis should have length of 3, and so on.
    :param unscaled: The computation does the operations using the natural logarithm log(). To obtain the actual entropy
                value one must scale this value by log(2) since the entropy should be computed in base 2 (hence log2()).
                However, one may desire using this function in a loss function to train a neural net. Then, the log(2)
                is just a multiplicative constant of the gradient and could be omitted for efficiency reasons. Turning
                this flag to True allows for this behaviour to happen (default is False, then the actual entropy).
    :param smooth: This small value will be added to the numerator and denominator.
    :return:
    """

    assert incoming.dtype in [tf.float32, tf.float64]

    # compute probability of label l
    p_l = tf.reduce_sum(incoming, axis=axis)
    p_l = tf.clip_by_value(p_l, clip_value_min=smooth, clip_value_max=1-smooth)
    entropy_l = - p_l * tf.log(p_l) - (1 - p_l) * tf.log(1 - p_l)

    if not unscaled:
        entropy_l = tf.log(2.0) * entropy_l

    entropy = tf.reduce_sum(entropy_l, axis=-1)
    mean_entropy = tf.reduce_mean(entropy, axis=0)

    return mean_entropy
