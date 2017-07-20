#! /usr/bin/env python

import numpy as np
import tensorflow as tf

from data_helpers import data_helpers_allAspect_glove

# Parameters
# ==================================================

aspect_name = ["All", "Value", "Room", "Location", "Cleanliness", "Service"]

# Eval Parameters
dir_code = 1469509124
parent_dir = "./runs/final_oneAspect/"
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", parent_dir + str(dir_code) + "/checkpoints/",
                       "Checkpoint directory from training run")

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
x, y, vocabulary, vocabulary_inv, s_count = data_helpers_allAspect_glove.load_test_data()
y_overall = y[:, 2, :]
# y_test = np.argmax(y_test, axis=1)
print("Vocabulary size: {:d}".format(len(vocabulary)))
print("Test set size {:d}".format(len(y_overall)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print "latest check point file" + str(checkpoint_file)
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
        input_y_all = graph.get_operation_by_name("input_y_all").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        # accuracy_1 = graph.get_operation_by_name("accuracy/aspect_1/accuracy-1").outputs[0]
        # accuracy_2 = graph.get_operation_by_name("accuracy/aspect_2/accuracy-2").outputs[0]
        # accuracy_3 = graph.get_operation_by_name("accuracy/aspect_3/accuracy-3").outputs[0]
        # accuracy_4 = graph.get_operation_by_name("accuracy/aspect_4/accuracy-4").outputs[0]
        # accuracy_5 = graph.get_operation_by_name("accuracy/aspect_5/accuracy-5").outputs[0]

        all_rating = graph.get_operation_by_name("output/overall_value").outputs[0]

        # Generate batches for one epoch
        x_batches = data_helpers_allAspect_glove.batch_iter(x, FLAGS.batch_size, 1, shuffle=False)
        y_batches_all = data_helpers_allAspect_glove.batch_iter(y_overall, FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_prediction = []
        review_index = 0
        for x_test_batch, y_test_batch_all in zip(x_batches, y_batches_all):
            # rate_result
            # [aspect review]
            # sent_relate
            # [batch_size * sentence, num_aspects]
            overall_result = sess.run([all_rating],
                                      {input_x: x_test_batch,
                                       input_y_all: y_test_batch_all,
                                       dropout_keep_prob: 1.0})

            all_prediction.append(overall_result)

        all_prediction = np.concatenate(all_prediction, axis=1)

np.savetxt(parent_dir + str(dir_code) + "/all_rating.out", all_prediction + 1, fmt='%1.0f')

y_overall_value = np.argmax(y_overall, axis=1)

print("Total number of test examples: {}".format(len(y_overall)))

correct_predictions = all_prediction == y_overall_value
accuracy = np.mean(correct_predictions)
print "accuracy\toverall\t" + str(accuracy)
mse = np.mean((all_prediction - y_overall_value) ** 2)
print "MSE\toverall\t" + str(mse) + "\n"
