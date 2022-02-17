import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import importlib
import sys


pd.set_option('display.max_columns', 500)

sys.path.append('/content/Adversarially-Learned-Anomaly-Detection')
from utils.adapt_data import batch_fill
from utils.evaluations import save_results, heatmap, plot_log
from utils.constants import IMAGES_DATASETS

FREQ_PRINT = 200  # print frequency image tensorboard [20]
FREQ_EV = 1
PATIENCE = 10


def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter


def display_parameters(batch_size, starting_lr, ema_decay, degree, label,
                       allow_zz, score_method, do_spectral_norm,nb_epochs):
    """See parameters
    """
    print("Number of Epochs: ",nb_epochs)
    print('Batch size: ', batch_size)
    # print('Starting learning rate: ', starting_lr)
    # print('EMA Decay: ', ema_decay)
    print('Degree for L norms: ', degree)
    # print('Anomalous label: ', label)
    # print('Score method: ', score_method)
    print('Discriminator zz enabled: ', allow_zz)
    # print('Spectral Norm enabled: ', do_spectral_norm)


def display_progression_epoch(j, id_max):
    """See epoch progression
    """
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def create_logdir(dataset, label, rd,
                  allow_zz, score_method, do_spectral_norm):
    """ Directory to save training logs, weights, biases, etc."""
    model = 'alad_sn{}_dzz{}'.format(do_spectral_norm, allow_zz)
    return "../../train_logs/{}_{}_dzzenabled{}_{}_label{}" \
           "rd{}".format(dataset, model, allow_zz,
                         score_method, label, rd)


def train_and_test(dataset, nb_epochs, degree, random_seed, label,
                   allow_zz, enable_sm, score_method,
                   enable_early_stop, do_spectral_norm):
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
        score_method (str): which metric to use for the ablation study
        enable_early_stop (bool): allow early stopping for determining the number of epochs
        do_spectral_norm (bool): allow spectral norm or not for ablation study
     """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Import model and data
    sys.path.append('/content/Adversarially-Learned-Anomaly-Detection')
    network = importlib.import_module('alad.{}_utilities'.format(dataset))
    data = importlib.import_module("data.{}".format(dataset))

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.999

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Placeholders
    x_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input_x")
    z_pl = tf.placeholder(tf.float32, shape=[None, latent_dim], name="input_z")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    print('Data loading...')
    trainx, trainy = data.get_train(label)
    trainx_copy = trainx.copy()
    if enable_early_stop: validx, validy = data.get_valid(label)
    testx, testy = data.get_test(label)
    print(trainx.shape)
    rng = np.random.RandomState(random_seed)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    print('Building graph...')
    print("ALAD is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, degree, label,
                       allow_zz, score_method, do_spectral_norm,nb_epochs)

    gen = network.decoder
    enc = network.encoder
    dis_xz = network.discriminator_xz
    dis_xx = network.discriminator_xx
    dis_zz = network.discriminator_zz
    dis_xxzz = network.discriminator_xxzz

    with tf.variable_scope('encoder_model'):
        z_gen = enc(x_pl, is_training=is_training_pl,
                    do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('generator_model'):
        x_gen = gen(z_pl, is_training=is_training_pl)
        rec_x = gen(z_gen, is_training=is_training_pl, reuse=True)

    with tf.variable_scope('encoder_model'):
        rec_z = enc(x_gen, is_training=is_training_pl, reuse=True,
                    do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('discriminator_model_xz'):
        l_encoder, inter_layer_inp_xz = dis_xz(x_pl, z_gen,
                                               is_training=is_training_pl,
                                               do_spectral_norm=do_spectral_norm)
        l_generator, inter_layer_rct_xz = dis_xz(x_gen, z_pl,
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
        z_logit_real, _ = dis_zz(z_pl, z_pl, is_training=is_training_pl,
                                 do_spectral_norm=do_spectral_norm)
        z_logit_fake, _ = dis_zz(z_pl, rec_z, is_training=is_training_pl,
                                 reuse=True, do_spectral_norm=do_spectral_norm)

    with tf.variable_scope('discriminator_model_xxzz'):
        xz_logit_real, _ = dis_xxzz(x_pl, x_pl, z_pl, z_pl, is_training=is_training_pl,
                                    do_spectral_norm=do_spectral_norm)
        xz_logit_fake, _ = dis_xxzz(x_pl, rec_x, z_pl, rec_z, is_training=is_training_pl,
                                    reuse=True, do_spectral_norm=do_spectral_norm)

    with tf.name_scope('loss_functions'):

        # discriminator xz
        loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(l_encoder), logits=l_encoder))
        loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(l_generator), logits=l_generator))
        dis_loss_xz = loss_dis_gen + loss_dis_enc

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
        cost_xz = 0  # tf.reduce_mean(xz_real_gen + xz_fake_gen)

        cycle_consistency_loss = cost_x + cost_z + cost_xz if allow_zz else cost_x + cost_xz
        loss_generator = gen_loss_xz + cycle_consistency_loss
        loss_encoder = enc_loss_xz + cycle_consistency_loss

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
    with tf.variable_scope('discriminator_model_xxzz'):
        l_encoder_emaxxzz, inter_layer_inp_emaxxzz = dis_xxzz(x_pl, x_pl, z_gen_ema, z_gen_ema,
                                                              is_training=is_training_pl,
                                                              getter=get_getter(xx_ema),
                                                              reuse=True,
                                                              do_spectral_norm=do_spectral_norm)

        l_generator_emaxxzz, inter_layer_rct_emaxxzz = dis_xxzz(x_pl, rec_x_ema, z_gen_ema, z_pl,
                                                                is_training=is_training_pl,
                                                                getter=get_getter(xx_ema),
                                                                reuse=True,
                                                                do_spectral_norm=do_spectral_norm)
    log_loss_dis = []
    with tf.name_scope('Testing'):

        with tf.variable_scope('Scores'):
            score_ch = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(l_generator_emaxx),
                logits=l_generator_emaxx)
            score_ch = tf.squeeze(score_ch)

            rec = x_pl - rec_x_ema
            rec = tf.layers.flatten(rec)
            score_l1 = tf.norm(rec, ord=1, axis=1,
                               keep_dims=False, name='d_loss')
            score_l1 = tf.squeeze(score_l1)

            rec = x_pl - rec_x_ema
            rec = tf.layers.flatten(rec)
            score_l2 = tf.norm(rec, ord=2, axis=1,
                               keep_dims=False, name='d_loss')
            score_l2 = tf.squeeze(score_l2)

            inter_layer_inp, inter_layer_rct = inter_layer_inp_emaxx, \
                                               inter_layer_rct_emaxx
            fm = inter_layer_inp - inter_layer_rct
            fm = tf.layers.flatten(fm)
            score_fm = tf.norm(fm, ord=degree, axis=1,
                               keep_dims=False, name='d_loss')
            score_fm = tf.squeeze(score_fm)

            score_z_ema = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(l_generator_emaxxzz),
                logits=l_generator_emaxxzz)
            score_z_ema = tf.squeeze(score_z_ema)

            score_z = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(xz_logit_fake),
                logits=xz_logit_fake)
            score_z = tf.squeeze(score_z)

    if enable_early_stop:
        rec_error_valid = tf.reduce_mean(score_fm)

    logdir = create_logdir(dataset, label, random_seed, allow_zz, score_method,
                           do_spectral_norm)

    saver = tf.train.Saver(max_to_keep=2)
    save_model_secs = None  # if enable_early_stop else 20
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None, saver=saver, save_model_secs=save_model_secs)

    print('Start training...')
    with sv.managed_session(config=config) as sess:

        step = sess.run(global_step)
        # print('Initialization done at step {}'.format(step / nr_batches_train))
        # writer = tf.summary.FileWriter(logdir, sess.graph)
        train_batch = 0
        epoch = 0
        best_valid_loss = 0
        request_stop = False

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
                             z_pl: np.random.normal(size=[batch_size, latent_dim]),
                             is_training_pl: True,
                             learning_rate: lr}

                _, _, _, ld, ldxz, ldxx, ldzz, ldxxzz, step = sess.run([train_dis_op_xz,
                                                                        train_dis_op_xx,
                                                                        train_dis_op_zz,
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

            # if allow_zz:
            #     print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | "
            #           "loss dis = %.4f | loss dis xz = %.4f | loss dis xx = %.4f | "
            #           "loss dis zz = %.4f | loss dis xxzz = %.4f |"
            #           % (epoch, time.time() - begin, train_loss_gen,
            #              train_loss_enc, train_loss_dis, train_loss_dis_xz,
            #              train_loss_dis_xx, train_loss_dis_zz, train_loss_dis_xxzz))

            # else:
            #     print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | "
            #           "loss dis = %.4f | loss dis xz = %.4f | loss dis xx = %.4f | loss dis xxzz = %.4f | "
            #           % (epoch, time.time() - begin, train_loss_gen,
            #              train_loss_enc, train_loss_dis, train_loss_dis_xz,
            #              train_loss_dis_xx, train_loss_dis_xxzz))

            log_loss_dis.append(train_loss_dis)
            ##EARLY STOPPING
            if (epoch + 1) % FREQ_EV == 0 and enable_early_stop:

                valid_loss = 0
                feed_dict = {x_pl: validx,
                             z_pl: np.random.normal(size=[validx.shape[0], latent_dim]),
                             is_training_pl: False}
                vl, lat = sess.run([rec_error_valid, rec_z], feed_dict=feed_dict)
                valid_loss += vl

                print('Validation: valid loss {:.4f}'.format(valid_loss))

                if (valid_loss < best_valid_loss or epoch == FREQ_EV - 1):
                    best_valid_loss = valid_loss
                    print("Best model - valid loss = {:.4f} - saving...".format(best_valid_loss))
                    # sv.saver.save(sess, logdir + '/model.ckpt', global_step=step)
                    nb_without_improvements = 0
                else:
                    nb_without_improvements += FREQ_EV

                if nb_without_improvements > PATIENCE:
                    sv.request_stop()
                    print(
                        "Early stopping at epoch {} with weights from epoch {}".format(
                            epoch, epoch - nb_without_improvements))

            epoch += 1

        # sv.saver.save(sess, logdir + '/model.ckpt', global_step=step)

        print('Testing evaluation...')
        scores_z_ema = []
        scores_z = []
        scores_ch = []
        scores_l1 = []
        scores_l2 = []
        scores_fm = []
        inference_time = []

        # Create scores
        for t in range(nr_batches_test):
            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_test_time_batch = time.time()

            feed_dict = {x_pl: testx[ran_from:ran_to],
                         z_pl: np.random.normal(size=[batch_size, latent_dim]),
                         is_training_pl: False}
            scores_z_ema += sess.run(score_z_ema, feed_dict=feed_dict).tolist()
            scores_z += sess.run(score_z, feed_dict=feed_dict).tolist()
            scores_ch += sess.run(score_ch, feed_dict=feed_dict).tolist()
            scores_l1 += sess.run(score_l1, feed_dict=feed_dict).tolist()
            scores_l2 += sess.run(score_l2, feed_dict=feed_dict).tolist()
            scores_fm += sess.run(score_fm, feed_dict=feed_dict).tolist()
            inference_time.append(time.time() - begin_test_time_batch)

        inference_time = np.mean(inference_time)
        print('Testing : mean inference time is %.4f' % (inference_time))

        if testx.shape[0] % batch_size != 0:
            batch, size = batch_fill(testx, batch_size)
            feed_dict = {x_pl: batch,
                         z_pl: np.random.normal(size=[batch_size, latent_dim]),
                         is_training_pl: False}
            bscores_z_ema = sess.run(score_z_ema, feed_dict=feed_dict).tolist()
            bscores_z = sess.run(score_z, feed_dict=feed_dict).tolist()
            bscores_ch = sess.run(score_ch, feed_dict=feed_dict).tolist()
            bscores_l1 = sess.run(score_l1, feed_dict=feed_dict).tolist()
            bscores_l2 = sess.run(score_l2, feed_dict=feed_dict).tolist()
            bscores_fm = sess.run(score_fm, feed_dict=feed_dict).tolist()
            scores_z_ema += bscores_z_ema[:size]
            scores_z += bscores_z[:size]
            scores_ch += bscores_ch[:size]
            scores_l1 += bscores_l1[:size]
            scores_l2 += bscores_l2[:size]
            scores_fm += bscores_fm[:size]

        model = 'alad_sn{}_dzz{}'.format(do_spectral_norm, allow_zz)
        result_z_ema = save_results(scores_z_ema, testy, model, dataset, 'z_ema',
                                    'dzzenabled{}'.format(allow_zz), label, random_seed, step)
        result_z = save_results(scores_z, testy, model, dataset, 'z',
                                'dzzenabled{}'.format(allow_zz), label, random_seed, step)
        result_ch = save_results(scores_ch, testy, model, dataset, 'ch',
                                 'dzzenabled{}'.format(allow_zz), label, random_seed, step)
        result_l1 = save_results(scores_l1, testy, model, dataset, 'l1',
                                 'dzzenabled{}'.format(allow_zz), label, random_seed, step)
        result_l2 = save_results(scores_l2, testy, model, dataset, 'l2',
                                 'dzzenabled{}'.format(allow_zz), label, random_seed, step)
        result_fm = save_results(scores_fm, testy, model, dataset, 'fm',
                                 'dzzenabled{}'.format(allow_zz), label, random_seed, step)
        # plot_log(log_loss_dis,"loss discriminator")
        return result_z_ema, result_z, result_ch, result_l1, result_l2, result_fm


def run(args):
    """ Runs the training process"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(args.rd)
        train_and_test(args.dataset, args.nb_epochs, args.d, args.rd, args.label,
                       args.enable_dzz, args.enable_sm, args.m,
                       args.enable_early_stop, args.sn)


def describe_result(type_score,results):
    print("Describe Result for ",type_score," scoring")
    df_results = pd.DataFrame(results,columns=['precision','recall','f1','roc_auc'])
    print(df_results.describe(include='all')[1:3])
    print("-------------------------------------------")


results_z_ema, results_z, results_ch, results_l1, results_l2, results_fm = [],[],[],[],[],[]
for random_seed in range(10):
    print("===========================================")
    print("start round ",random_seed)
    tf.keras.backend.clear_session()
    tf.reset_default_graph()
    tf.Graph().as_default()
    # tf.set_random_seed(random_seed)
    result_z_ema, result_z, result_ch, result_l1, result_l2, result_fm = \
        train_and_test(dataset="arrhythmia", nb_epochs=1000, degree=2, random_seed=2
                       , label=1, allow_zz=True, enable_sm=True, score_method=""
                       , enable_early_stop=False, do_spectral_norm=False)
    results_z_ema.append(result_z_ema)
    results_z.append(result_z)
    results_ch.append(result_ch)
    results_l1.append(result_l1)
    results_l2.append(result_l2)
    results_fm.append(result_fm)

describe_result('z_ema',results_z_ema)
describe_result('z',results_z)
describe_result('ch',results_ch)
describe_result('l1',results_l1)
describe_result('l2',results_l2)
describe_result('fm',results_fm)