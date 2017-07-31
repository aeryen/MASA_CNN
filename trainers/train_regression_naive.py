#! /usr/bin/env python

import datetime
import os
import sys
import time

import numpy as np
import tensorflow as tf

from data_helpers import data_helpers_allAspect
from networks.cnn_regress_naive import RegressCNN

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("model_name", "naive regression", "regression model with 6 linear output number based off cnn")

tf.flags.DEFINE_integer("num_classes", 5, "Number of possible labels")
tf.flags.DEFINE_integer("num_aspects", 6, "Number of aspects")

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(("{}={}".format(attr.upper(), value)))
print("")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")

# if not os.path.isfile("xyvocab.pickle"):
time1 = time.time()
x, y, vocabulary, vocabulary_inv = data_helpers_allAspect.load_data()
time2 = time.time()
print('Load Data took ' + str((time2 - time1) * 1000.0) + ' ms')
print('x have a size of ' + str(sys.getsizeof(x) / 1024.0) + 'kbytes')

# pickle.dump([x, y, vocabulary, vocabulary_inv], open("xyvocab.pickle", "wb"))
# else:
#     x, y, vocabulary, vocabulary_inv = pickle.load(open("xyvocab.pickle", "rb"))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
print(("Vocabulary Size: {:d}".format(len(vocabulary))))
print(("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev))))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = RegressCNN(
            num_sentence=x_train.shape[1],
            num_word=x_train.shape[2],
            num_aspects=FLAGS.num_aspects,
            num_classes=FLAGS.num_classes,  # Number of classification classes
            vocab_size=len(vocabulary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        with tf.name_scope('grad_summary'):
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print(("Writing to {}\n".format(out_dir)))
        hyper_param_file = open(os.path.join(out_dir, 'param.txt'), 'w')
        hyper_param_file.write("\nParameters:")
        for attr, value in sorted(FLAGS.__flags.items()):
            hyper_param_file.write("{}={}\n".format(attr.upper(), value))
        hyper_param_file.close()

        # Summaries for loss and accuracy
        # global loss
        loss_summary = tf.scalar_summary("loss", cnn.loss)

        # per aspect MSE
        aspect_MSE = []
        for i in range(FLAGS.num_aspects):
            aspect_MSE.append(tf.scalar_summary("aspect_mse/aspect_" + str(i), cnn.mse_aspects[i]))

        # per label percentage for debug
        pred_ratio_summary = []
        for i in range(FLAGS.num_classes):
            pred_ratio_summary.append(
                tf.scalar_summary("prediction/label_" + str(i) + "_percentage", cnn.rate_percentage[i]))

        # overall accuracy ?
        acc_summary = tf.scalar_summary("average-accuracy", cnn.accuracy)

        # Train Summaries
        with tf.name_scope('train_summary'):
            train_summary_op = tf.merge_summary([loss_summary, aspect_MSE, pred_ratio_summary, acc_summary,
                                                 grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        with tf.name_scope('dev_summary'):
            dev_summary_op = tf.merge_summary([loss_summary, aspect_MSE, pred_ratio_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, all_mse = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.mse_aspects[0]],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print(("{}: step {}, loss {:g}, acc {:g}, mse {:g}".format(time_str, step, loss, accuracy, all_mse)))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1
            }
            step, summaries, loss, accuracy, all_mse, one_mse, two_mse, three_mse = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.mse_aspects[0],
                 cnn.mse_aspects[1], cnn.mse_aspects[2], cnn.mse_aspects[3]],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print(("\t{}: step {}, loss {:g}, acc {:g}\n"
                  "\t\tmse_all {:g}, mse_a1 {:g}, mse_a2 {:g}, mse_a3 {:g}".
                  format(time_str, step, loss, accuracy,
                         all_mse, one_mse, two_mse, three_mse)
                  ))
            if writer:
                writer.add_summary(summaries, step)


        # Generate batches
        batches = data_helpers_allAspect.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = list(zip(*batch))
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_batches = data_helpers_allAspect.batch_iter(list(zip(x_dev, y_dev)), 64, 1)
                for dev_batch in dev_batches:
                    small_dev_x, small_dev_y = list(zip(*dev_batch))
                    dev_step(small_dev_x, small_dev_y, writer=dev_summary_writer)
                    print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print(("Saved model checkpoint to {}\n".format(path)))
