#! /usr/bin/env python

import numpy as np
import tensorflow as tf

import data_helpers
from data_helpers import data_helpers_oneAspect_glove

# Parameters4000
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/original cnn redo/1470130862/checkpoints/",
                       "Checkpoint directory from training run")
aspect_id = 5
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data. Load your own data here
print("Loading data...")
x_test, y_test, vocabulary, vocabulary_inv = data_helpers_oneAspect_glove.load_test_data(aspect_id)
y_test = np.argmax(y_test, axis=1)
print("Vocabulary size: {:d}".format(len(vocabulary)))
print("Test set size {:d}".format(len(y_test)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        x_batches = data_helpers.batch_iter(x_test, FLAGS.batch_size, 1, shuffle=False)
        y_batches = data_helpers.batch_iter(y_test, FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch, y_test_batch in zip(x_batches, y_batches):
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            # batch_predictions = sess.run([accuracy, accuracy_1, accuracy_2, accuracy_3, accuracy_4, accuracy_5],
            #                              {input_x: x_test_batch, input_y: y_test_batch, dropout_keep_prob: 1.0})
            print batch_predictions
            # all_predictions.append(batch_predictions)
            all_predictions = np.concatenate([all_predictions, batch_predictions], axis=0)

# Print accuracy
np.savetxt('temp.out', all_predictions, fmt='%1.0f')
correct_predictions = float(sum(all_predictions == y_test))
# all_predictions = np.array(all_predictions)
# average_accuracy = all_predictions.sum(axis=0) / float(all_predictions.shape[0])
average_accuracy = correct_predictions / float(all_predictions.shape[0])
print("Total number of test examples: {}".format(len(y_test)))
print "\t" + str(average_accuracy)

mse = np.mean((all_predictions - y_test) ** 2)
print "MSE\t" + str(mse)
# print("Accuracy: {:g}".format(average_accuracy / float(len(y_test))))
