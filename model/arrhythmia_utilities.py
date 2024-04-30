# """

# Arrhythmia RCALAD architecture.

# Generator (decoder), encoder and discriminator.

# """
# # import tensorflow.compat.v1 as tf
# from tensorflow import compat
# tf = compat.v1

# tf.disable_eager_execution()
# tf.disable_v2_behavior()

# learning_rate = 1e-5
# batch_size = 32
# latent_dim = 64
# init_kernel = tf.truncated_normal_initializer()


# class sn:
#     def conv2d(inputs, filters, kernel_size, strides=1, padding='valid',
#                use_bias=True, kernel_initializer=None,
#                bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
#                name=None, reuse=None):

#         with tf.variable_scope(name, reuse=reuse):
#             w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, inputs.get_shape()[-1], filters],
#                                 initializer=kernel_initializer,
#                                 regularizer=kernel_regularizer)
#             bias = tf.get_variable("bias", [filters], initializer=bias_initializer)
#             x = tf.nn.conv2d(input=inputs, filter=sn.spectral_norm(w),
#                              strides=[1, strides, strides, 1], padding=padding)
#             if use_bias:
#                 x = tf.nn.bias_add(x, bias)

#         return x

#     def dense(inputs, units, use_bias=True, kernel_initializer=None,
#               bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
#               name=None, reuse=None):

#         with tf.variable_scope(name, reuse=reuse):
#             inputs = tf.layers.flatten(inputs)
#             shape = inputs.get_shape().as_list()
#             channels = shape[-1]

#             w = tf.get_variable("kernel", [channels, units], tf.float32,
#                                 initializer=kernel_initializer, regularizer=kernel_regularizer)
#             if use_bias:
#                 bias = tf.get_variable("bias", [units],
#                                        initializer=bias_initializer)

#                 x = tf.matmul(inputs, sn.spectral_norm(w)) + bias
#             else:
#                 x = tf.matmul(inputs, sn.spectral_norm(w))

#         return x

#     def spectral_norm(w, iteration=1, eps=1e-12):
#         w_shape = w.shape.as_list()
#         w = tf.reshape(w, [-1, w_shape[-1]])

#         u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

#         u_hat = u
#         v_hat = None
#         for i in range(iteration):
#             """
#             power iteration
#             Usually iteration = 1 will be enough
#             """
#             v_ = tf.matmul(u_hat, tf.transpose(w))
#             v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + eps)

#             u_ = tf.matmul(v_hat, w)
#             u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + eps)

#         sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
#         w_norm = w / sigma

#         with tf.control_dependencies([u.assign(u_hat)]):
#             w_norm = tf.reshape(w_norm, w_shape)

#         return w_norm


# def leakyReLu(x, alpha=0.2, name=None):
#     if name:
#         with tf.variable_scope(name):
#             return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))
#     else:
#         return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


# def encoder(x_inp, is_training=False, getter=None, reuse=False,
#             do_spectral_norm=False):
#     """ Encoder architecture in tensorflow

#     Maps the data into the latent space

#     Args:
#         x_inp (tensor): input data for the encoder.
#         is_training (bool): for batch norms and dropouts
#         getter: for exponential moving average during inference
#         reuse (bool): sharing variables or not

#     Returns:
#         net (tensor): last activation layer of the encoder

#     """
#     with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):
#         name_net = 'layer_1'
#         with tf.variable_scope(name_net):
#             net = tf.layers.dense(x_inp,
#                                   units=256,
#                                   kernel_initializer=init_kernel,
#                                   name='fc')
#             net = leakyReLu(net)

#         name_net = 'layer_2'
#         with tf.variable_scope(name_net):
#             net = tf.layers.dense(net,
#                                   units=128,
#                                   kernel_initializer=init_kernel,
#                                   name='fc')
#             net = leakyReLu(net)

#         name_net = 'layer_3'
#         with tf.variable_scope(name_net):
#             net = tf.layers.dense(net,
#                                   units=latent_dim,
#                                   kernel_initializer=init_kernel,
#                                   name='fc')
#     return net


# def decoder(z_inp, is_training=False, getter=None, reuse=False):
#     """ Generator architecture in tensorflow

#     Generates data from the latent space

#     Args:
#         z_inp (tensor): input variable in the latent space
#         is_training (bool): for batch norms and dropouts
#         getter: for exponential moving average during inference
#         reuse (bool): sharing variables or not

#     Returns:
#         net (tensor): last activation layer of the generator

#     """
#     with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):
#         name_net = 'layer_1'
#         with tf.variable_scope(name_net):
#             net = tf.layers.dense(z_inp,
#                                   units=128,
#                                   kernel_initializer=init_kernel,
#                                   name='fc')
#             net = tf.nn.relu(net)

#         name_net = 'layer_2'
#         with tf.variable_scope(name_net):
#             net = tf.layers.dense(net,
#                                   units=256,
#                                   kernel_initializer=init_kernel,
#                                   name='fc')
#             net = tf.nn.relu(net)

#         name_net = 'layer_3'
#         with tf.variable_scope(name_net):
#             net = tf.layers.dense(net,
#                                   units=274,
#                                   kernel_initializer=init_kernel,
#                                   name='fc')
#     return net


# def discriminator_xz(x_inp, z_inp, is_training=False, getter=None, reuse=False,
#                      do_spectral_norm=False):
#     """ Discriminator architecture in tensorflow

#     Discriminates between pairs (E(x), x) and (z, G(z))

#     Args:
#         x_inp (tensor): input data for the discriminator.
#         z_inp (tensor): input variable in the latent space
#         is_training (bool): for batch norms and dropouts
#         getter: for exponential moving average during inference
#         reuse (bool): sharing variables or not

#     Returns:
#         logits (tensor): last activation layer of the discriminator (shape 1)
#         intermediate_layer (tensor): intermediate layer for feature matching

#     """
#     with tf.variable_scope('discriminator_xz', reuse=reuse, custom_getter=getter):
#         # D(x)
#         name_x = 'x_layer_1'
#         with tf.variable_scope(name_x):
#             x = tf.layers.dense(x_inp,
#                                 units=128,
#                                 kernel_initializer=init_kernel,
#                                 name='fc')
#             x = tf.layers.batch_normalization(x,
#                                               training=is_training,
#                                               name='batch_normalization')
#             x = leakyReLu(x)

#         # D(z)
#         name_z = 'z_layer_1'
#         with tf.variable_scope(name_z):
#             z = tf.layers.dense(z_inp, 128, kernel_initializer=init_kernel)
#             z = leakyReLu(z)
#             z = tf.layers.dropout(z, rate=0.5, name='dropout', training=is_training)

#         # D(x,z)
#         y = tf.concat([x, z], axis=1)

#         name_y = 'y_layer_1'
#         with tf.variable_scope(name_y):
#             y = tf.layers.dense(y,
#                                 256,
#                                 kernel_initializer=init_kernel)
#             y = leakyReLu(y)
#             y = tf.layers.dropout(y, rate=0.5, name='dropout', training=is_training)

#         intermediate_layer = y

#         name_y = 'y_layer_2'
#         with tf.variable_scope(name_y):
#             logits = tf.layers.dense(y,
#                                      1,
#                                      kernel_initializer=init_kernel)

#     return logits, intermediate_layer


# def discriminator_xxzz(x, rec_x, z, rec_z, is_training=False, getter=None, reuse=False,
#                        do_spectral_norm=False):
#     """ Discriminator architecture in tensorflow

#     Discriminates between pairs (x, x, E(x), E(x)) and ( x , G(z),z , E(x))

#     Args:
#         x (tensor): input from the data space
#         rec_x (tensor): reconstructed data
#         z (tensor): input from the latent space
#         rec_z (tensor): reconstructed data
#         is_training (bool): for batch norms and dropouts
#         getter: for exponential moving average during inference
#         reuse (bool): sharing variables or not

#     Returns:
#         logits (tensor): last activation layer of the discriminator (shape 1)
#         intermediate_layer (tensor): intermediate layer for feature matching

#     """
#     with tf.variable_scope('discriminator_xxzz', reuse=reuse, custom_getter=getter):
#         # D(x,x)
#         name_x = 'xx_layer_1'
#         net_x = tf.concat([x, rec_x], axis=1)
#         with tf.variable_scope(name_x):
#             x = tf.layers.dense(net_x,
#                                 units=256,
#                                 kernel_initializer=init_kernel,
#                                 name='fc')
#             x = tf.layers.batch_normalization(x,
#                                               training=is_training,
#                                               name='batch_normalization')
#             x = leakyReLu(x)

#         name_net = 'xx_layer_2'
#         with tf.variable_scope(name_net):
#             x = tf.layers.dense(x,
#                                   units=128,
#                                   kernel_initializer=init_kernel,
#                                   name='fc')
#             x = leakyReLu(x)
#             x = tf.layers.dropout(x, rate=0.2, name='dropout', training=is_training)

#         # D(z,z)
#         name_z = 'zz_layer_1'
#         net_z = tf.concat([z, rec_z], axis=1)
#         with tf.variable_scope(name_z):
#             z = tf.layers.dense(net_z, 64, kernel_initializer=init_kernel)
#             z = leakyReLu(z)
#             z = tf.layers.dropout(z, rate=0.5, name='dropout', training=is_training)

#         # D(x,x,z,z)
#         y = tf.concat([x, z], axis=1)

#         name_y = 'y_layer_1'
#         with tf.variable_scope(name_y):
#             y = tf.layers.dense(y,
#                                 128,
#                                 kernel_initializer=init_kernel)
#             y = leakyReLu(y)
#             y = tf.layers.dropout(y, rate=0.5, name='dropout', training=is_training)

#         intermediate_layer = y
#         name_y = 'y_layer_2'
#         with tf.variable_scope(name_y):
#             y = tf.layers.dense(y,
#                                 32,
#                                 kernel_initializer=init_kernel)
#         name_y = 'y_layer_3'
#         with tf.variable_scope(name_y):
#             logits = tf.layers.dense(y,
#                                      1,
#                                      kernel_initializer=init_kernel)

#     return logits, intermediate_layer


# def discriminator_xx(x, rec_x, is_training=False, getter=None, reuse=False,
#                      do_spectral_norm=False):
#     """ Discriminator architecture in tensorflow

#     Discriminates between (x,x) and (x,rec_x)

#     Args:
#         x (tensor): input from the data space
#         rec_x (tensor): reconstructed data
#         is_training (bool): for batch norms and dropouts
#         getter: for exponential moving average during inference
#         reuse (bool): sharing variables or not

#     Returns:
#         logits (tensor): last activation layer of the discriminator
#         intermediate_layer (tensor): intermediate layer for feature matching

#     """
#     with tf.variable_scope('discriminator_xx', reuse=reuse, custom_getter=getter):
#         net = tf.concat([x, rec_x], axis=1)

#         name_net = 'layer_1'
#         with tf.variable_scope(name_net):
#             net = tf.layers.dense(net,
#                                   units=256,
#                                   kernel_initializer=init_kernel,
#                                   name='fc')
#             net = leakyReLu(net)
#             net = tf.layers.dropout(net, rate=0.2, name='dropout', training=is_training)

#         name_net = 'layer_2'
#         with tf.variable_scope(name_net):
#             net = tf.layers.dense(net,
#                                   units=128,
#                                   kernel_initializer=init_kernel,
#                                   name='fc')
#             net = leakyReLu(net)
#             net = tf.layers.dropout(net, rate=0.2, name='dropout', training=is_training)

#         intermediate_layer = net

#         name_net = 'layer_3'
#         with tf.variable_scope(name_net):
#             logits = tf.layers.dense(net,
#                                      units=1,
#                                      kernel_initializer=init_kernel,
#                                      name='fc')

#     return logits, intermediate_layer


# def discriminator_zz(z, rec_z, is_training=False, getter=None, reuse=False,
#                      do_spectral_norm=False):
#     """ Discriminator architecture in tensorflow

#     Discriminates between (z,z) and (z,rec_z)

#     Args:
#         z (tensor): input from the latent space
#         rec_z (tensor): reconstructed data
#         is_training (bool): for batch norms and dropouts
#         getter: for exponential moving average during inference
#         reuse (bool): sharing variables or not

#     Returns:
#         logits (tensor): last activation layer of the discriminator
#         intermediate_layer (tensor): intermediate layer for feature matching

#     """
#     with tf.variable_scope('discriminator_zz', reuse=reuse, custom_getter=getter):
#         net = tf.concat([z, rec_z], axis=-1)
#         name_net = 'layer_1'
#         with tf.variable_scope(name_net):
#             net = tf.layers.dense(net,
#                                   units=64,
#                                   kernel_initializer=init_kernel,
#                                   name='fc')

#             net = leakyReLu(net, 0.2, name='conv1/leaky_relu')
#             net = tf.layers.dropout(net, rate=0.2, name='dropout',
#                                     training=is_training)

#         name_net = 'layer_2'
#         with tf.variable_scope(name_net):
#             net = tf.layers.dense(net,
#                                   units=32,
#                                   kernel_initializer=init_kernel,
#                                   name='fc')

#             net = leakyReLu(net, 0.2, name='conv1/leaky_relu')
#             net = tf.layers.dropout(net, rate=0.2, name='dropout',
#                                     training=is_training)
#         intermediate_layer = net

#         name_net = 'layer_3'
#         with tf.variable_scope(name_net):
#             logits = tf.layers.dense(net,
#                                      units=1,
#                                      kernel_initializer=init_kernel,
#                                      name='fc')

#     return logits, intermediate_layer


from tensorflow import keras
from tensorflow import compat
tf = compat.v1

tf.disable_v2_behavior()

learning_rate = 1e-5
batch_size = 32
latent_dim = 64
init_kernel = keras.initializers.TruncatedNormal()


class sn:
    def conv2d(inputs, filters, kernel_size, strides=1, padding='valid',
               use_bias=True, kernel_initializer=None,
               bias_initializer=keras.initializers.Zeros(), kernel_regularizer=None,
               name=None, reuse=None):

        with tf.variable_scope(name, reuse=reuse):
            w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, inputs.get_shape()[-1], filters],
                                initializer=kernel_initializer,
                                regularizer=kernel_regularizer)
            bias = tf.get_variable("bias", [filters], initializer=bias_initializer)
            x = tf.nn.conv2d(input=inputs, filter=sn.spectral_norm(w),
                             strides=[1, strides, strides, 1], padding=padding)
            if use_bias:
                x = tf.nn.bias_add(x, bias)

        return x

    def dense(inputs, units, use_bias=True, kernel_initializer=None,
              bias_initializer=keras.initializers.Zeros(), kernel_regularizer=None,
              name=None, reuse=None):

        with tf.variable_scope(name, reuse=reuse):
            flatten = keras.layers.Flatten()(inputs)
            dense_layer = keras.layers.Dense(units, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer)
            return dense_layer(flatten)

    def spectral_norm(w, iteration=1, eps=1e-12):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + eps)

            u_ = tf.matmul(v_hat, w)
            u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + eps)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm


def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))
    else:
        return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


def encoder(x_inp, is_training=False, getter=None, reuse=False,
            do_spectral_norm=False):
    with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):
        net = keras.Sequential([
            keras.layers.Dense(256, activation='relu', kernel_initializer=init_kernel),
            keras.layers.Dense(128, activation='relu', kernel_initializer=init_kernel),
            keras.layers.Dense(latent_dim, kernel_initializer=init_kernel)
        ])
        return net(x_inp)


def decoder(z_inp, is_training=False, getter=None, reuse=False):
    with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):
        net = keras.Sequential([
            keras.layers.Dense(128, activation='relu', kernel_initializer=init_kernel),
            keras.layers.Dense(256, activation='relu', kernel_initializer=init_kernel),
            keras.layers.Dense(274, kernel_initializer=init_kernel)
        ])
        return net(z_inp)

def discriminator_xz(x_inp, z_inp, is_training=False, getter=None, reuse=False,
                     do_spectral_norm=False):
    """ Discriminator architecture in tensorflow """
    with tf.variable_scope('discriminator_xz', reuse=reuse, custom_getter=getter):
        # D(x)
        x = keras.layers.Dense(128, kernel_initializer=init_kernel)(x_inp)
        x = keras.layers.BatchNormalization()(x, training=is_training)
        x = leakyReLu(x)

        # D(z)
        z = keras.layers.Dense(128, kernel_initializer=init_kernel)(z_inp)
        z = leakyReLu(z)
        z = keras.layers.Dropout(0.5)(z, training=is_training)

        # D(x,z)
        y = tf.concat([x, z], axis=1)
        y = keras.layers.Dense(256, kernel_initializer=init_kernel)(y)
        y = leakyReLu(y)
        y = keras.layers.Dropout(0.5)(y, training=is_training)

        intermediate_layer = y
        logits = keras.layers.Dense(1, kernel_initializer=init_kernel)(y)

    return logits, intermediate_layer

def discriminator_xxzz(x, rec_x, z, rec_z, is_training=False, getter=None, reuse=False,
                       do_spectral_norm=False):
    """ Discriminator architecture in tensorflow """
    with tf.variable_scope('discriminator_xxzz', reuse=reuse, custom_getter=getter):
        # D(x,x)
        net_x = tf.concat([x, rec_x], axis=1)
        x = keras.layers.Dense(256, kernel_initializer=init_kernel)(net_x)
        x = keras.layers.BatchNormalization()(x, training=is_training)
        x = leakyReLu(x)

        # D(z,z)
        net_z = tf.concat([z, rec_z], axis=1)
        z = keras.layers.Dense(64, kernel_initializer=init_kernel)(net_z)
        z = leakyReLu(z)
        z = keras.layers.Dropout(0.5)(z, training=is_training)

        # D(x,x,z,z)
        y = tf.concat([x, z], axis=1)
        y = keras.layers.Dense(128, kernel_initializer=init_kernel)(y)
        y = leakyReLu(y)
        y = keras.layers.Dropout(0.5)(y, training=is_training)

        intermediate_layer = y
        y = keras.layers.Dense(32, kernel_initializer=init_kernel)(y)
        logits = keras.layers.Dense(1, kernel_initializer=init_kernel)(y)

    return logits, intermediate_layer

def discriminator_xx(x, rec_x, is_training=False, getter=None, reuse=False,
                     do_spectral_norm=False):
    """ Discriminator architecture in tensorflow """
    with tf.variable_scope('discriminator_xx', reuse=reuse, custom_getter=getter):
        net = tf.concat([x, rec_x], axis=1)
        net = keras.layers.Dense(256, kernel_initializer=init_kernel)(net)
        net = leakyReLu(net)
        net = keras.layers.Dropout(0.2)(net, training=is_training)

        net = keras.layers.Dense(128, kernel_initializer=init_kernel)(net)
        net = leakyReLu(net)
        net = keras.layers.Dropout(0.2)(net, training=is_training)

        intermediate_layer = net
        logits = keras.layers.Dense(1, kernel_initializer=init_kernel)(net)

    return logits, intermediate_layer

def discriminator_zz(z, rec_z, is_training=False, getter=None, reuse=False,
                     do_spectral_norm=False):
    """ Discriminator architecture in tensorflow """
    with tf.variable_scope('discriminator_zz', reuse=reuse, custom_getter=getter):
        net = tf.concat([z, rec_z], axis=-1)
        net = keras.layers.Dense(64, kernel_initializer=init_kernel)(net)
        net = leakyReLu(net, 0.2)
        net = keras.layers.Dropout(0.2)(net, training=is_training)

        net = keras.layers.Dense(32, kernel_initializer=init_kernel)(net)
        net = leakyReLu(net, 0.2)
        net = keras.layers.Dropout(0.2)(net, training=is_training)

        intermediate_layer = net
        logits = keras.layers.Dense(1, kernel_initializer=init_kernel)(net)

    return logits, intermediate_layer
