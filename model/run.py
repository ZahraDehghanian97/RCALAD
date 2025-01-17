import warnings
import pandas as pd

warnings.filterwarnings('ignore')
import time
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)
import os

IMAGES_DATASETS = ['cifar10', 'svhn']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import importlib
import sys

pd.set_option('display.max_columns', 500)

sys.path.append('/content/RCALAD')
from utils.adapt_data import batch_fill
from utils.evaluations import save_results, heatmap, plot_log
from utils.constants import IMAGES_DATASETS

FREQ_PRINT = 200  # print frequency image tensorboard [20]
FREQ_EV = 1


def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter


def display_parameters(batch_size, starting_lr, ema_decay, degree, label,
                       allow_zz, do_spectral_norm, nb_epochs):
    """See parameters
    """
    print("Number of Epochs: ", nb_epochs)
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Degree for L norms: ', degree)
    print('Anomalous label: ', label)
    print('Spectral Norm enabled: ', do_spectral_norm)


def display_progression_epoch(j, id_max):
    """See epoch progression
    """
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def create_logdir(dataset, label, rd,
                  allow_zz, do_spectral_norm):
    """ Directory to save training logs, weights, biases, etc."""
    model = 'RCALAD_sn{}_dzz{}'.format(do_spectral_norm, allow_zz)
    return "../../train_logs/{}_{}_dzzenabled{}_label{}" \
           "rd{}".format(dataset, model, allow_zz, label, rd)


def train_and_test(dataset, nb_epochs, degree, random_seed, label,
                   allow_zz, do_spectral_norm):
    """
    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        dataset (str): name of the dataset
        nb_epochs (int): number of epochs
        degree (int): degree of the norm in the feature matching
        random_seed (int): trying different seeds for averaging the results
        label (int): label which is normal for image experiments
        allow_zz (bool): allow the d_zz discriminator or not for ablation study
        enable_sm (bool): allow TF summaries for monitoring the training
        do_spectral_norm (bool): allow spectral norm or not for ablation study
     """
    global alpha, beta
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Import model and data
    sys.path.append('/content/RCALAD')
    network = importlib.import_module('model.{}_utilities'.format(dataset))
    data = importlib.import_module("data.{}".format(dataset))

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    x_dim = data.get_shape_input()
    ema_decay = 0.999

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Placeholders
    x_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input_x")
    x_pl_t = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input_x_t")
    z_pl = tf.placeholder(tf.float32, shape=[None, latent_dim], name="input_z")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    print('Data loading...')
    trainx, trainy = data.get_train(label)
    trainx_copy = trainx.copy()
    testx, testy = data.get_test(label)
    trainx_t = np.random.normal(size=trainx.shape)
    testx_t = np.random.normal(size=testx.shape)
    print(trainx.shape)
    rng = np.random.RandomState(random_seed)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    print('Building graph...')
    print("RCALAD is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, degree, label,
                       allow_zz, do_spectral_norm, nb_epochs)

    gen = network.decoder
    enc = network.encoder
    dis_xz = network.discriminator_xz
    dis_xx = network.discriminator_xx
    dis_zz = network.discriminator_zz
    dis_xxzz = network.discriminator_xxzz

    with tf.variable_scope('encoder_model'):
        z_gen = enc(x_pl, is_training=is_training_pl,
                    do_spectral_norm=do_spectral_norm)
        z_gen_t = enc(x_pl_t, is_training=is_training_pl,
                      do_spectral_norm=do_spectral_norm, reuse=True)

    with tf.variable_scope('generator_model'):
        x_gen = gen(z_pl, is_training=is_training_pl)
        rec_x = gen(z_gen, is_training=is_training_pl, reuse=True)

    with tf.variable_scope('encoder_model'):
        rec_z = enc(x_gen, is_training=is_training_pl, reuse=True,
                    do_spectral_norm=do_spectral_norm)
        rec_z_gen = enc(rec_x, is_training=is_training_pl,
                      do_spectral_norm=do_spectral_norm, reuse=True)

    with tf.variable_scope('discriminator_model_xz'):
        l_encoder, inter_layer_inp_xz = dis_xz(x_pl, z_gen,
                                               is_training=is_training_pl,
                                               do_spectral_norm=do_spectral_norm)
        l_generator, inter_layer_rct_xz = dis_xz(x_gen, z_pl,
                                                 is_training=is_training_pl,
                                                 reuse=True,
                                                 do_spectral_norm=do_spectral_norm)
        l_t, _ = dis_xz(x_pl_t, z_gen_t,
                      is_training=is_training_pl,
                      reuse=True,
                      do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('discriminator_model_xx'):
        x_logit_real, inter_layer_inp_xx = dis_xx(x_pl, x_pl,
                                                  is_training=is_training_pl,
                                                  do_spectral_norm=do_spectral_norm)
        x_logit_fake, inter_layer_rct_xx = dis_xx(x_pl, rec_x, is_training=is_training_pl,
                                                  reuse=True, do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('discriminator_model_zz'):
        z_logit_real, inter_layer_inp_zz = dis_zz(z_pl, z_pl, is_training=is_training_pl,
                                                  do_spectral_norm=do_spectral_norm)
        z_logit_fake, inter_layer_rct_zz = dis_zz(z_pl, rec_z, is_training=is_training_pl,
                                                  reuse=True, do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('discriminator_model_xxzz'):
        xz_logit_real, inter_layer_inp_xxzz = dis_xxzz(x_pl, x_pl, z_gen, z_gen, is_training=is_training_pl,
                                                       do_spectral_norm=do_spectral_norm)
        xz_logit_fake, inter_layer_rct_xxzz = dis_xxzz(x_pl, rec_x, z_gen, rec_z_gen, is_training=is_training_pl,
                                                       reuse=True, do_spectral_norm=do_spectral_norm)

    with tf.name_scope('loss_functions'):

        # discriminator xz
        loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(l_encoder), logits=l_encoder))
        loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(l_generator), logits=l_generator))
        loss_dis_t = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(l_t), logits=l_t))
        dis_loss_xz = loss_dis_gen + loss_dis_enc + loss_dis_t

        # discriminator xx
        x_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_real, labels=tf.ones_like(x_logit_real))
        x_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_fake, labels=tf.zeros_like(x_logit_fake))
        dis_loss_xx = tf.reduce_mean(x_real_dis + x_fake_dis)

        # discriminator zz
        z_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_real, labels=tf.ones_like(z_logit_real))
        z_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_fake, labels=tf.zeros_like(z_logit_fake))
        dis_loss_zz = tf.reduce_mean(z_real_dis + z_fake_dis)

        # discriminator xxzz
        xz_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=xz_logit_real, labels=tf.ones_like(xz_logit_real))
        xz_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=xz_logit_fake, labels=tf.zeros_like(xz_logit_fake))
        dis_loss_xxzz = tf.reduce_mean(xz_real_dis + xz_fake_dis)

        loss_discriminator = dis_loss_xz + dis_loss_xx + dis_loss_zz + dis_loss_xxzz if \
            allow_zz else dis_loss_xz + dis_loss_xx + dis_loss_xxzz

        # generator and encoder
        gen_loss_xz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(l_generator), logits=l_generator))
        enc_loss_xz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(l_encoder), logits=l_encoder))
        x_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_real, labels=tf.zeros_like(x_logit_real))
        x_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit_fake, labels=tf.ones_like(x_logit_fake))
        z_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_real, labels=tf.zeros_like(z_logit_real))
        z_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=z_logit_fake, labels=tf.ones_like(z_logit_fake))
        xz_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=xz_logit_real, labels=tf.zeros_like(xz_logit_real))
        xz_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=xz_logit_fake, labels=tf.ones_like(xz_logit_fake))

        cost_x = tf.reduce_mean(x_real_gen + x_fake_gen)
        cost_z = tf.reduce_mean(z_real_gen + z_fake_gen)
        cost_xz = tf.reduce_mean(xz_real_gen + xz_fake_gen)/10

        cycle_consistency_loss = cost_x + cost_z + cost_xz if allow_zz else cost_x + cost_xz
        loss_generator = gen_loss_xz  + cycle_consistency_loss
        loss_encoder = enc_loss_xz  + cycle_consistency_loss

    with tf.name_scope('optimizers'):

        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dxzvars = [var for var in tvars if 'discriminator_model_xz' in var.name]
        dxxvars = [var for var in tvars if 'discriminator_model_xx' in var.name]
        dzzvars = [var for var in tvars if 'discriminator_model_zz' in var.name]
        dxxzzvars = [var for var in tvars if 'discriminator_model_xxzz' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis_xz = [x for x in update_ops if
                             ('discriminator_model_xz' in x.name)]
        update_ops_dis_xx = [x for x in update_ops if
                             ('discriminator_model_xx' in x.name)]
        update_ops_dis_zz = [x for x in update_ops if
                             ('discriminator_model_zz' in x.name)]
        update_ops_dis_xxzz = [x for x in update_ops if
                               ('discriminator_model_xxzz' in x.name)]

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.5)

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer.minimize(loss_generator, var_list=gvars,
                                        global_step=global_step)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer.minimize(loss_encoder, var_list=evars)

        with tf.control_dependencies(update_ops_dis_xz):
            dis_op_xz = optimizer.minimize(dis_loss_xz, var_list=dxzvars)

        with tf.control_dependencies(update_ops_dis_xx):
            dis_op_xx = optimizer.minimize(dis_loss_xx, var_list=dxxvars)

        with tf.control_dependencies(update_ops_dis_zz):
            dis_op_zz = optimizer.minimize(dis_loss_zz, var_list=dzzvars)

        with tf.control_dependencies(update_ops_dis_xxzz):
            dis_op_xxzz = optimizer.minimize(dis_loss_xxzz, var_list=dxxzzvars)

        # Exponential Moving Average for inference
        def train_op_with_ema_dependency(vars, op):
            ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
            maintain_averages_op = ema.apply(vars)
            with tf.control_dependencies([op]):
                train_op = tf.group(maintain_averages_op)
            return train_op, ema

        train_gen_op, gen_ema = train_op_with_ema_dependency(gvars, gen_op)
        train_enc_op, enc_ema = train_op_with_ema_dependency(evars, enc_op)
        train_dis_op_xz, xz_ema = train_op_with_ema_dependency(dxzvars,
                                                               dis_op_xz)
        train_dis_op_xx, xx_ema = train_op_with_ema_dependency(dxxvars,
                                                               dis_op_xx)
        train_dis_op_zz, zz_ema = train_op_with_ema_dependency(dzzvars,
                                                               dis_op_zz)
        train_dis_op_xxzz, xxzz_ema = train_op_with_ema_dependency(dxxzzvars,
                                                                   dis_op_xxzz)

    with tf.variable_scope('encoder_model'):
        z_gen_ema = enc(x_pl, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True,
                        do_spectral_norm=do_spectral_norm)
    with tf.variable_scope('generator_model'):
        rec_x_ema = gen(z_gen_ema, is_training=is_training_pl,
                        getter=get_getter(gen_ema), reuse=True)
        x_gen_ema = gen(z_pl, is_training=is_training_pl,
                        getter=get_getter(gen_ema), reuse=True)
    with tf.variable_scope('encoder_model'):
        rec_z_ema = enc(x_gen_ema, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True,
                        do_spectral_norm=do_spectral_norm)
        rec_z_gen_ema = enc(rec_x_ema, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True,
                        do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('discriminator_model_xx'):
        l_encoder_emaxx, inter_layer_inp_emaxx = dis_xx(x_pl, x_pl,
                                                        is_training=is_training_pl,
                                                        getter=get_getter(xx_ema),
                                                        reuse=True,
                                                        do_spectral_norm=do_spectral_norm)

        l_generator_emaxx, inter_layer_rct_emaxx = dis_xx(x_pl, rec_x_ema,
                                                          is_training=is_training_pl,
                                                          getter=get_getter(
                                                              xx_ema),
                                                          reuse=True,
                                                          do_spectral_norm=do_spectral_norm)
    with tf.variable_scope('discriminator_model_zz'):
        l_encoder_emazz, inter_layer_inp_emazz = dis_zz(z_pl, z_pl,
                                                        is_training=is_training_pl,
                                                        getter=get_getter(zz_ema),
                                                        reuse=True,
                                                        do_spectral_norm=do_spectral_norm)

        l_generator_emazz, inter_layer_rct_emazz = dis_zz(z_pl, rec_z_ema,
                                                          is_training=is_training_pl,
                                                          getter=get_getter(
                                                              zz_ema),
                                                          reuse=True,
                                                          do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('discriminator_model_xxzz'):
        l_encoder_emaxxzz, inter_layer_inp_emaxxzz = dis_xxzz(x_pl, x_pl, z_gen_ema, z_gen_ema,
                                                              is_training=is_training_pl,
                                                              getter=get_getter(xxzz_ema),
                                                              reuse=True,
                                                              do_spectral_norm=do_spectral_norm)

        l_generator_emaxxzz, inter_layer_rct_emaxxzz = dis_xxzz(x_pl, rec_x_ema, z_gen_ema,rec_z_gen_ema,
                                                                is_training=is_training_pl,
                                                                getter=get_getter(xxzz_ema),
                                                                reuse=True,
                                                                do_spectral_norm=do_spectral_norm)
    log_loss_dis = []
    with tf.name_scope('Testing'):

        with tf.variable_scope('Scores'):

            score_logits_dxx = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(l_generator_emaxx),
                logits=l_generator_emaxx)
            score_logits_dxx = tf.squeeze(score_logits_dxx)

            score_logits_dzz = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(l_generator_emazz),
                logits=l_generator_emazz)
            score_logits_dzz = tf.squeeze(score_logits_dzz)

            score_logits_dxxzz = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(l_generator_emaxxzz),
                logits=l_generator_emaxxzz)
            score_logits_dxxzz = tf.squeeze(score_logits_dxxzz)

            score_logits_all = score_logits_dxx + score_logits_dzz + score_logits_dxxzz

            inter_layer_inp, inter_layer_rct = inter_layer_inp_emaxxzz, \
                                               inter_layer_rct_emaxxzz
            fm = inter_layer_inp - inter_layer_rct
            fm = tf.layers.flatten(fm)
            score_fm_xxzz = tf.norm(fm, ord=degree, axis=1,
                                    keep_dims=False, name='d_loss')
            score_fm_xxzz = tf.squeeze(score_fm_xxzz)

    logdir = create_logdir(dataset, label, random_seed, allow_zz,
                           do_spectral_norm)

    saver = tf.train.Saver(max_to_keep=2)
    save_model_secs = None  # if enable_early_stop else 20
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None, saver=saver, save_model_secs=save_model_secs)

    print('Start training...')
    with sv.managed_session(config=config) as sess:

        step = sess.run(global_step)
        train_batch = 0
        epoch = 0

        while not sv.should_stop() and epoch < nb_epochs:

            lr = starting_lr
            begin = time.time()

            # construct randomly permuted minibatches
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]
            train_loss_dis_xz, train_loss_dis_xx, train_loss_dis_zz, train_loss_dis_xxzz, \
            train_loss_dis, train_loss_gen, train_loss_enc = [0, 0, 0, 0, 0, 0, 0]

            # Training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train discriminator
                feed_dict = {x_pl: trainx[ran_from:ran_to],
                            x_pl_t: trainx_t[ran_from:ran_to],
                             z_pl: np.random.normal(size=[batch_size, latent_dim]),
                             is_training_pl: True,
                             learning_rate: lr}

                _, _, _,_, ld, ldxz, ldxx, ldzz, ldxxzz, step = sess.run([train_dis_op_xz,
                                                                        train_dis_op_xx,
                                                                        train_dis_op_zz,
                                                                        train_dis_op_xxzz,
                                                                        loss_discriminator,
                                                                        dis_loss_xz,
                                                                        dis_loss_xx,
                                                                        dis_loss_zz,
                                                                        dis_loss_xxzz,
                                                                        global_step],
                                                                       feed_dict=feed_dict)
                train_loss_dis += ld
                train_loss_dis_xz += ldxz
                train_loss_dis_xx += ldxx
                train_loss_dis_zz += ldzz
                train_loss_dis_xxzz += ldxxzz

                # train generator and encoder
                feed_dict = {x_pl: trainx_copy[ran_from:ran_to],
                            x_pl_t: trainx_t[ran_from:ran_to],
                             z_pl: np.random.normal(size=[batch_size, latent_dim]),
                             is_training_pl: True,
                             learning_rate: lr}
                _, _, le, lg = sess.run([train_gen_op,
                                         train_enc_op,
                                         loss_encoder,
                                         loss_generator],
                                        feed_dict=feed_dict)
                train_loss_gen += lg
                train_loss_enc += le

                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_enc /= nr_batches_train
            train_loss_dis /= nr_batches_train
            train_loss_dis_xz /= nr_batches_train
            train_loss_dis_xx /= nr_batches_train
            train_loss_dis_zz /= nr_batches_train
            train_loss_dis_xxzz /= nr_batches_train
            if epoch % 10 == 0:
                if allow_zz:
                    print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | "
                          "loss dis = %.4f | loss dis xz = %.4f | loss dis xx = %.4f | "
                          "loss dis zz = %.4f | loss dis xxzz = %.4f |"
                          % (epoch, time.time() - begin, train_loss_gen,
                             train_loss_enc, train_loss_dis, train_loss_dis_xz,
                             train_loss_dis_xx, train_loss_dis_zz, train_loss_dis_xxzz))

                else:
                    print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | "
                          "loss dis = %.4f | loss dis xz = %.4f | loss dis xx = %.4f | loss dis xxzz = %.4f | "
                          % (epoch, time.time() - begin, train_loss_gen,
                             train_loss_enc, train_loss_dis, train_loss_dis_xz,
                             train_loss_dis_xx, train_loss_dis_xxzz))

            log_loss_dis.append(train_loss_dis)
            epoch += 1

        # sv.saver.save(sess, logdir + '/model.ckpt', global_step=step)

        print('Testing evaluation...')
        scores_logits_all = []
        scores_fm_xxzz = []
        inference_time = []

        # Create scores
        for t in range(nr_batches_test):
            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_test_time_batch = time.time()

            feed_dict = {x_pl: testx[ran_from:ran_to],
                        x_pl_t: testx_t[ran_from:ran_to],
                        z_pl: np.random.normal(size=[batch_size, latent_dim]),
                        is_training_pl: True,
                        learning_rate: lr}

            scores_fm_xxzz += sess.run(score_fm_xxzz, feed_dict=feed_dict).tolist()
            scores_logits_all += sess.run(score_logits_all, feed_dict=feed_dict).tolist()
            inference_time.append(time.time() - begin_test_time_batch)

        inference_time = np.mean(inference_time)
        print('Testing : mean inference time is %.4f' % (inference_time))

        if testx.shape[0] % batch_size != 0:
            batch, size = batch_fill(testx, batch_size)
            feed_dict = {x_pl: batch,
                        x_pl_t: testx_t[ran_from:ran_to],
                         z_pl: np.random.normal(size=[batch_size, latent_dim]),
                         is_training_pl: False}

            bscores_fm_xxzz = sess.run(score_fm_xxzz, feed_dict=feed_dict).tolist()
            bscores_logits_all = sess.run(score_logits_all, feed_dict=feed_dict).tolist()

            scores_fm_xxzz += bscores_fm_xxzz[:size]
            scores_logits_all += bscores_logits_all[:size]

        model = 'RCALAD_sn{}_dzz{}'.format(do_spectral_norm, allow_zz)
        result_fm_xxzz = save_results(scores_fm_xxzz, testy, model, dataset, 'dxxzz',
                                      'dzzenabled{}'.format(allow_zz), label, random_seed, step)
        result_logits_all = save_results(scores_logits_all, testy, model, dataset, 'd_all',
                                         'dzzenabled{}'.format(allow_zz), label, random_seed, step)

        # plot_log(log_loss_dis,"loss discriminator")
        return result_fm_xxzz, result_logits_all


def add_result(dataset,score_array, x, method):
    global IMAGES_DATASETS
    print("----------------------------")
    if dataset in IMAGES_DATASETS:
        print("Testing with method %s: AUROC = %.4f"
              % (method, x[3]))
    else:
        print("Testing with method %s: Prec = %.4f | Rec = %.4f | F1 = %.4f"
              % (method, x[0], x[1], x[2]))
    score_array.append(x)
    return score_array


def describe_result(type_score, results):
    print("-------------------------------------------")
    print("Describe Result for ", type_score, " scoring")
    df_results = pd.DataFrame(results, columns=['precision', 'recall', 'f1', 'roc_auc'])
    print(df_results)


def run(args):
    """ Runs the training process"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    results_fm_xxzz, results_logits_all = [], []
    print("===========================================")
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(args.rd)
        result_fm_xxzz, result_logits_all =train_and_test(dataset=args.dataset, nb_epochs=args.nb_epochs,
                       random_seed=args.rd,degree=args.d ,label= args.label, allow_zz=args.enable_dzz, do_spectral_norm= args.sn)

        results_fm_xxzz = add_result(args.dataset,results_fm_xxzz, result_fm_xxzz, "fm_xxzz")
        results_logits_all = add_result(args.dataset,results_logits_all, result_logits_all, "logits_all")

        describe_result('fm_xxzz', results_fm_xxzz)
        describe_result('logits_all', results_logits_all)

