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
tf.random.set_random_seed(1234)
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
from idas.utils import safe_mkdir
import PIL.Image
from idas.tf_utils import from_one_hot_to_rgb
from medpy.metric.binary import hd
import config
from idas.utils import print_yellow_text


args = config.define_flags()


def hausdorff_distance(mask1, mask2):
    """Compute the average Hausdorff distance for the patient (in pixels), between mask1 and mask2."""

    def _py_hd(m1, m2):
        """Python function to compute HD between the two n-dimensional masks"""
        m1, m2 = np.array(m1), np.array(m2)
        num_elems = len(m1)
        assert len(m2) == num_elems

        # remove last channel, if it is == 1:
        if len(m1.shape) == 4 and m1.shape[-1] == 1:
            m1, m2 = np.squeeze(m1, axis=-1), np.squeeze(m2, axis=-1)
        try:
            return hd(m1, m2)
        except:
            # maximum distance
            return min([m1.shape[1], m1.shape[2]])

    # map _py_hd(.) to every element on the batch axis:
    tf_hd = tf.py_function(func=_py_hd, inp=[mask1, mask2],
                           Tout=[tf.float32], name='hausdorff_distance')

    # return the average HD in the batch:
    return tf.reduce_mean(tf_hd)


def dice_coe(output, target, axis=(1, 2, 3), smooth=1e-12):
    """Compute the average Dice score between output and target segmentation masks."""
    intersection = tf.reduce_sum(output * target, axis=axis)
    a = tf.reduce_sum(output, axis=axis)
    b = tf.reduce_sum(target, axis=axis)
    score = (2. * intersection + smooth) / (a + b + smooth)
    score = tf.reduce_mean(score, name='dice_coe')
    return score


def iou_coe(output, target, axis=(1, 2, 3), smooth=1e-12):
    """Compute the average IOU score between output and target segmentation masks."""
    intersection = tf.reduce_sum(output * target, axis=axis)

    a = tf.reduce_sum(output * output, axis=axis)
    b = tf.reduce_sum(target * target, axis=axis)

    union = a + b - intersection
    score = (intersection + smooth) / (union + smooth)
    score = tf.reduce_mean(score)
    return score


def get_channel(incoming, idx):
    return tf.expand_dims(incoming[..., idx], -1)


def test_model(sess, model, n_images=3):
    """ Test the model once """
    # -------------------------------------------------------------------------------------------
    def _test_model(n_cls, n_img, pred, true):

        # global and class-specific metrics:
        dice_test_list = [dice_coe(output=pred[..., 1:], target=true[..., 1:])]
        dice_test_list_cls = [dice_coe(output=get_channel(pred, i), target=get_channel(true, i)) for i in range(n_cls)]
        iou_test_list = [iou_coe(output=pred[..., 1:], target=true[..., 1:])]
        iou_test_list_cls = [iou_coe(output=get_channel(pred, i), target=get_channel(true, i)) for i in range(n_cls)]
        hd_test_list_cls = [hausdorff_distance(pred[..., i], true[..., i]) for i in range(n_cls)]
        # hd_test_list is average(hd_test_list_cls)

        # global list of test to do:
        test_list = []
        test_list.extend(dice_test_list)
        test_list.extend(dice_test_list_cls)
        test_list.extend(iou_test_list)
        test_list.extend(iou_test_list_cls)
        test_list.extend(hd_test_list_cls)

        # assign results to each value
        if n_img > 0:
            # add also images to the test list
            image_list = [model.input, from_one_hot_to_rgb(pred, background='white'),
                          from_one_hot_to_rgb(true, background='white')]
            test_list.extend(image_list)

        # dc, dc_per_class, iu, iu_per_class, hd_per_class, img0, img1, img2 = \
        #     sess.run(test_list, feed_dict={model.is_training: False})
        # imgs_tuple = [img0, img1, img2]
        results = sess.run(test_list, feed_dict={model.is_training: False})
        idx = 0
        dc = results[idx]
        dc_per_class = []
        for c in range(n_cls):
            idx += 1
            dc_per_class.append(results[idx])

        idx += 1
        iu = results[idx]
        iu_per_class = []
        for c in range(n_cls):
            idx += 1
            iu_per_class.append(results[idx])

        # this time start from 0 as we don't have hd inside results
        hd_per_class = []
        for c in range(n_cls):
            idx += 1
            hd_per_class.append(results[idx])
        hd = np.mean(hd_per_class)
        if n_img > 0:
            idx += 1
            imgs_tuple = results[idx:]
        else:
            imgs_tuple = None
        return dc, dc_per_class, iu, iu_per_class, hd, hd_per_class, imgs_tuple

    # -------------------------------------------------------------------------------------------

    # Test
    n_classes = model.n_classes
    sess.run(model.test_init)  # initialize data set iterator on test set:
    y_pred = model.prediction  # model prediction
    y_true = model.ground_truth  # ground truth segmentation

    # initialize a dictionary with the metrics
    metrics = {'dice': dict(), 'iou': dict(), 'hd': dict()}
    metrics['dice']['global'] = list()
    metrics['iou']['global'] = list()
    metrics['hd']['global'] = list()
    for ch in range(n_classes):
        metrics['dice'][ch] = list()
        metrics['iou'][ch] = list()
        metrics['hd'][ch] = list()

    n_batches = 0
    img_list = []
    try:
        while True:
            dice, dice_per_class, iou, iou_per_class, hdist, hdist_per_class, imgs = \
                _test_model(n_classes, n_images, y_pred, y_true)

            n_batches += 1
            n_images -= 1

            # save results
            metrics['dice']['global'].append(dice)
            metrics['iou']['global'].append(iou)
            metrics['hd']['global'].append(hdist)  # hd = hausdorff_distance
            for ch in range(n_classes):
                metrics['dice'][ch].append(dice_per_class[ch])
                metrics['iou'][ch].append(iou_per_class[ch])
                metrics['hd'][ch].append(hdist_per_class[ch])
            if imgs is not None: img_list.append(imgs)

    except tf.errors.OutOfRangeError:
        # End of the test set. Compute statistics here:
        avg_dice, std_dice = np.mean(metrics['dice']['global']), np.std(metrics['dice']['global'])
        avg_dice_per_class = [np.mean(metrics['dice'][c]) for c in range(n_classes)]
        std_dice_per_class = [np.mean(metrics['dice'][c]) for c in range(n_classes)]
        avg_iou, std_iou = np.mean(metrics['iou']['global']), np.std(metrics['iou']['global'])
        avg_iou_per_class = [np.mean(metrics['iou'][c]) for c in range(n_classes)]
        std_iou_per_class = [np.mean(metrics['iou'][c]) for c in range(n_classes)]
        avg_hd, std_hd = np.mean(metrics['hd']['global']), np.std(metrics['hd']['global'])
        avg_hd_per_class = [np.mean(metrics['hd'][c]) for c in range(n_classes)]
        std_hd_per_class = [np.mean(metrics['hd'][c]) for c in range(n_classes)]

        dice_list_per_class = [metrics['dice'][ch] for ch in range(n_classes)]
        iou_list_per_class = [metrics['iou'][ch] for ch in range(n_classes)]
        hd_list_per_class = [metrics['hd'][ch] for ch in range(n_classes)]

    return \
        avg_dice, std_dice, [avg_dice_per_class, std_dice_per_class], metrics['dice']['global'], dice_list_per_class, \
        avg_iou, std_iou, [avg_iou_per_class, std_iou_per_class], metrics['iou']['global'], iou_list_per_class, \
        avg_hd, std_hd, [avg_hd_per_class, std_hd_per_class], metrics['hd']['global'], hd_list_per_class, img_list


def plot_batch(img_list, path_prefix):
    """Save batch of images tiled."""

    def _postprocess_image(img):
        """ from float range in about [-1, 1] to uint8 in [0, 255] """
        # rescale:
        img = img + abs(img.min())
        img = img / img.max()
        img = np.clip(255 * img, 0, 255)
        img = img.astype(np.uint8)
        return img

    def _safe_rgb(img):
        """ Converts grayscale image to rgb, if needed """
        if img.shape[-1] == 1:
            img = np.stack((np.squeeze(img, axis=-1),) * 3, axis=-1)
        return img

    def _tile(img_lst, n_rows):
        """Tile images for display."""
        x, yp, yt = img_lst
        n_cols = 3  # one for each: x, yp, yt
        assert x.shape == yp.shape
        assert x.shape == yt.shape
        h, w = x.shape[1], x.shape[2]

        # initialize and then fill empty array with input images:
        tiled = np.zeros((n_rows * h, n_cols * w, 3), dtype=x.dtype)
        for row_i in range(n_rows):
            x_i, yp_i, yt_i = x[row_i], yp[row_i], yt[row_i]
            tiled[row_i * h: (row_i + 1) * h, 0: w, :] = x_i
            tiled[row_i * h: (row_i + 1) * h, w: 2 * w, :] = yp_i
            tiled[row_i * h: (row_i + 1) * h, 2 * w:, :] = yt_i
        return tiled

    for i in range(len(img_list)):
        x_in, y_pred, y_true = img_list[i]
        x_in = _postprocess_image(x_in)
        x_in = _safe_rgb(x_in)

        rows = len(x_in)
        canvas = _tile([x_in, y_pred, y_true], rows)
        canvas = np.squeeze(canvas)
        PIL.Image.fromarray(canvas).save(os.path.join(path_prefix, 'test_batch{0}.png'.format(i)))


# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================


def test(model, sess):
    print_yellow_text('Performing final test...')

    epoch = sess.run(model.g_epoch)

    # do a test:
    avg_dice, std_dice, [avg_dice_per_class, std_dice_per_class], dice_list, dice_list_per_class, \
        avg_iou, std_iou, [avg_iou_per_class, std_iou_per_class], iou_list, iou_list_per_class, \
        avg_hd, std_hd, [avg_hd_per_class, std_hd_per_class], hd_list, hd_list_per_class, \
        img_list = test_model(sess, model, n_images=6)

    print(f'Dice average, standard deviation: \t {avg_dice} \t {std_dice}')
    print(f'IoU average, standard deviation: \t {avg_iou} \t {std_iou}')
    print(f'HD average, standard deviation: \t {avg_hd} \t {std_hd}')


    # save the images:
    results_dir = args.results_dir
    dataset_name = args.dataset_name
    safe_mkdir('{0}/results/{1}/{2}/images/'.format(results_dir, args.experiment_type, dataset_name))
    safe_mkdir('{0}/results/{1}/{2}/images/{3}'.format(results_dir, args.experiment_type, dataset_name, args.n_sup_vols))
    safe_mkdir('{0}/results/{1}/{2}/images/{3}/{4}'.format(results_dir, args.experiment_type, dataset_name, args.n_sup_vols, model.run_id))
    image_dest_path = '{0}/results/{1}/{2}/images/{3}/{4}'.format(results_dir, args.experiment_type, dataset_name, args.n_sup_vols, model.run_id)
    plot_batch(img_list, path_prefix=image_dest_path)

    print(f'\nSaving test images under: {image_dest_path}')
