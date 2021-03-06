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
from tensorflow import layers

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class UNetPyAG(object):

    def __init__(self, n_out, is_training, n_filters=64, n_out_att=None, use_bn=True, upsampling_mode='NN',
                 name='UNetPyAG'):
        """
        Class for Hierarchical UNet architecture. This is a 2D version (hence the vanilla UNet), which means it only
        employs bi-dimensional convolution and strides. This implementation also uses batch normalization after each
        convolutional layer. The network, include PyAGs modules, detailed in the paper:

            > Valvano G., Leo A. and Tsaftaris S. A. (DART, 2021),
                Self-supervised Multi-scale Consistency for Weakly Supervised Segmentation Learning*.

        """
        assert upsampling_mode in ['deconv', 'NN']

        # network parameters
        self.n_out = n_out
        self.is_training = is_training
        self.nf = n_filters
        self.use_bn = use_bn
        self.upsampling_mode = upsampling_mode
        self.name = name

        if n_out_att is None:
            self.n_out_att = n_out
        else:
            self.n_out_att = n_out_att

        # final prediction
        self.prediction = None

    def build(self, incoming, reuse=tf.AUTO_REUSE):
        """
        Build the model.
        """
        with tf.variable_scope(self.name, reuse=reuse):
            encoder = self.build_encoder(incoming)
            code = self.build_bottleneck(encoder)
            decoder = self.build_decoder(code)
            self.prediction = self.build_output(decoder)

        return self

    def build_encoder(self, incoming):
        """ Encoder layers """

        # check for compatible input dimensions
        shape = incoming.get_shape().as_list()
        assert not shape[1] % 16
        assert not shape[2] % 16

        with tf.variable_scope('Encoder'):
            en_brick_0, concat_0 = self._encode_brick(incoming, self.nf, self.is_training,
                                                      scope='encode_brick_0', use_bn=self.use_bn)
            en_brick_1, concat_1 = self._encode_brick(en_brick_0, 2 * self.nf, self.is_training,
                                                      scope='encode_brick_1', use_bn=self.use_bn)
            en_brick_2, concat_2 = self._encode_brick(en_brick_1, 4 * self.nf, self.is_training,
                                                      scope='encode_brick_2', use_bn=self.use_bn)
            en_brick_3, concat_3 = self._encode_brick(en_brick_2, 8 * self.nf, self.is_training,
                                                      scope='encode_brick_3', use_bn=self.use_bn)

        return en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, en_brick_3, concat_3

    def build_bottleneck(self, encoder):
        """ Central layers """
        en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, en_brick_3, concat_3 = encoder

        with tf.variable_scope('Bottleneck'):
            code = self._bottleneck_brick(en_brick_3, 16 * self.nf, self.is_training, scope='code', use_bn=self.use_bn)

        return en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, en_brick_3, concat_3, code

    def build_decoder(self, code):
        """ Decoder layers """
        en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, en_brick_3, concat_3, code = code

        with tf.variable_scope('Decoder'):
            dec_brick_0, attention_0 = self._decode_brick(code, concat_3, 8 * self.nf, self.is_training,
                                                              scope='decode_brick_0', use_bn=self.use_bn)
            dec_brick_1, attention_1 = self._decode_brick(dec_brick_0, concat_2, 4 * self.nf, self.is_training,
                                                              scope='decode_brick_1', use_bn=self.use_bn)
            dec_brick_2, attention_2 = self._decode_brick(dec_brick_1, concat_1, 2 * self.nf, self.is_training,
                                                              scope='decode_brick_2', use_bn=self.use_bn)
            dec_brick_3, attention_3 = self._decode_brick(dec_brick_2, concat_0, self.nf, self.is_training,
                                                              scope='decode_brick_3', use_bn=self.use_bn)

        self.attention_tensors = [attention_2, attention_1, attention_0]

        return dec_brick_3

    def build_output(self, decoder):
        """ Output layers """
        # output linear
        return self._output_layer(decoder, n_channels_out=self.n_out, scope='output')

    @staticmethod
    def _encode_brick(incoming, nb_filters, is_training, scope, use_bn=True, trainable=True):
        """ Encoding brick: conv --> conv --> max pool.
        """
        with tf.variable_scope(scope):
            conv1 = layers.conv2d(incoming, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)

            if use_bn:
                conv1 = layers.batch_normalization(conv1, training=is_training, trainable=trainable)
            conv1_act = tf.nn.relu(conv1)

            conv2 = layers.conv2d(conv1_act, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)

            if use_bn:
                conv2 = layers.batch_normalization(conv2, training=is_training, trainable=trainable)
            conv2_act = tf.nn.relu(conv2)

            pool = layers.max_pooling2d(conv2_act, pool_size=2, strides=2, padding='same')

            with tf.variable_scope('concat_layer_out'):
                concat_layer_out = conv2_act
        return pool, concat_layer_out

    def _decode_brick(self, incoming, concat_layer_in, nb_filters, is_training, scope, use_bn=True):
        """ Decoding brick: deconv (up-pool) --> conv --> conv.
        """
        with tf.variable_scope(scope):

            if self.upsampling_mode == 'deconv':
                upsampled = layers.conv2d_transpose(incoming, filters=nb_filters, kernel_size=2, strides=2,
                                                    padding='same', kernel_initializer=he_init, bias_initializer=b_init)
                conv1 = upsampled
            else:
                # NN : nearest neighbor interpolation + conv
                _, old_height, old_width, __ = incoming.get_shape()
                new_height, new_width = 2.0 * old_height, 2.0 * old_width
                upsampled = tf.image.resize_nearest_neighbor(incoming, size=[new_height, new_width], align_corners=True)

                conv1 = layers.conv2d(upsampled, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                      kernel_initializer=he_init, bias_initializer=b_init)
            if use_bn:
                conv1 = layers.batch_normalization(conv1, training=is_training)
            # conv1_act = tf.nn.relu(conv1)
            conv1_act = conv1  # first without activation

            concat = tf.concat([conv1_act, concat_layer_in], axis=-1)

            conv2 = layers.conv2d(concat, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            if use_bn:
                conv2 = layers.batch_normalization(conv2, training=is_training)
            conv2_act = tf.nn.relu(conv2)

            conv3 = layers.conv2d(conv2_act, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            if use_bn:
                conv3 = layers.batch_normalization(conv3, training=is_training)
            conv3_act = tf.nn.relu(conv3)

            # PyAG attention module
            with tf.variable_scope('AttentionMap'):
                attention = layers.conv2d(conv2_act, filters=self.n_out_att, kernel_size=1, strides=1, padding='same',
                                          kernel_initializer=he_init, bias_initializer=b_init)
                attention = tf.nn.softmax(attention)
                attention_map = tf.reduce_sum(attention[..., 1:], axis=-1, keepdims=True)

            conv_and_att = conv3_act * attention_map

        return conv_and_att, attention

    @staticmethod
    def _bottleneck_brick(incoming, nb_filters, is_training, scope, use_bn=True, trainable=True):
        """ Code brick: conv --> conv .
        """
        with tf.variable_scope(scope):
            code1 = layers.conv2d(incoming, filters=nb_filters, kernel_size=1, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            if use_bn:
                code1 = layers.batch_normalization(code1, training=is_training, trainable=trainable)
            code1_act = tf.nn.relu(code1)

            code2 = layers.conv2d(code1_act, filters=nb_filters, kernel_size=1, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            if use_bn:
                code2 = layers.batch_normalization(code2, training=is_training, trainable=trainable)
            code2_act = tf.nn.relu(code2)

        return code2_act

    @staticmethod
    def _output_layer(incoming, n_channels_out, scope):
        """ Output layer: conv .
        """
        with tf.variable_scope(scope):
            output = layers.conv2d(incoming, filters=n_channels_out, kernel_size=1, strides=1, padding='same',
                                   kernel_initializer=he_init, bias_initializer=b_init)
            # activation = linear
        return output

    def get_prediction(self, one_hot=False, softmax=False):
        if one_hot:
            return tf.one_hot(tf.argmax(self.prediction, axis=-1), self.n_out)
        if softmax:
            return tf.nn.softmax(self.prediction)
        return self.prediction
