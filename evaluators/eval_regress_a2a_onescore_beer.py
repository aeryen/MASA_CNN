#! /usr/bin/env python

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

import data_helpers
from data_helpers import data_helpers_allAspect_glove_beer

# Parameters
# ==================================================

aspect_name = ["overall", "appearance", "taste", "palate", "aroma"]

# Eval Parameters
dir_code = "1484152848_100"
out_dir = "./runs/my_beer/regression2-2/"
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", out_dir + str(dir_code) + "/checkpoints/",
                       "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(("{}={}".format(attr.upper(), value)))
print("")

# Load data. Load your own data here
print("Loading data...")
x_test, y_test, vocabulary, vocabulary_inv, s_count = data_helpers_allAspect_glove_beer.load_test_data()
# y_test = np.argmax(y_test, axis=1)
print(("Vocabulary size: {:d}".format(len(vocabulary))))
print(("Test set size {:d}".format(len(y_test))))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print("latest check point file" + str(checkpoint_file))

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
        input_sc = graph.get_operation_by_name("input_s_count").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        batch_review_aspect_score = graph.get_operation_by_name("output/review_final_value").outputs[0]
        sent_relate_softmax = graph.get_operation_by_name("related/softmax_related").outputs[0]
        aspect_sway = graph.get_operation_by_name("rating-score/aspect/score_asp").outputs[0]

        # Generate batches for one epoch
        x_batches = data_helpers.batch_iter(x_test, FLAGS.batch_size, 1, shuffle=False)
        y_batches = data_helpers.batch_iter(y_test, FLAGS.batch_size, 1, shuffle=False)
        sc_batches = data_helpers.batch_iter(s_count, FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_prediction = []
        all_dist_aspect = []
        all_max_aspect = []
        all_sway_values = []
        review_index = 0
        for x_test_batch, y_test_batch, sc_test_batch in zip(x_batches, y_batches, sc_batches):
            # rate_result
            # [aspect review]
            # sent_relate
            # [batch_size * sentence, num_aspects]
            [review_final_value, sent_relate, aspect_s] = sess.run([batch_review_aspect_score,
                                                                    sent_relate_softmax,
                                                                    aspect_sway],
                                                                   {input_x: x_test_batch, input_y: y_test_batch,
                                                                    input_sc: sc_test_batch,
                                                                    dropout_keep_prob: 1.0})
            all_prediction.append(review_final_value)

            sent_relate_max = tf.argmax(sent_relate, 1).eval()
            for sent_index in range(review_final_value.shape[0]):
                all_dist_aspect.append(sent_relate[sent_index * 100:sent_index * 100 + s_count[review_index]])
                all_max_aspect.append(sent_relate_max[sent_index * 100:sent_index * 100 + s_count[review_index]])
                all_sway_values.append(aspect_s[sent_index * 100:sent_index * 100 + s_count[review_index]])
                review_index += 1

y_value = np.argmax(y_test, axis=2)

all_prediction = np.concatenate(all_prediction, axis=0)
all_dist_aspect = np.concatenate(all_dist_aspect, axis=0)
all_max_aspect = np.concatenate(all_max_aspect, axis=0)
all_sway_values = np.concatenate(all_sway_values, axis=0)
sentence_aspect_names = [aspect_name[i] for i in all_max_aspect]

np.savetxt(out_dir + str(dir_code) + '/aspect_rating.out', all_prediction, fmt='%1.2f')
np.savetxt(out_dir + str(dir_code) + '/aspect_dist.out', all_dist_aspect, fmt='%1.5f')
np.savetxt(out_dir + str(dir_code) + '/aspect_related.out', all_max_aspect, fmt='%1.0f')
np.savetxt(out_dir + str(dir_code) + '/aspect_sway.out', all_sway_values, fmt='%1.2f')

aspect_name_file = open(out_dir + str(dir_code) + '/aspect_related_name.out', 'w')
for item in sentence_aspect_names:
    aspect_name_file.write("%s\n" % item)
aspect_name_file.close()

all_prediction = all_prediction / 2.0
y_value = y_value / 2.0

print(("Total number of test examples: {}".format(len(y_test))))
for aspect_index in range(len(aspect_name)):
    correct_predictions = np.round(all_prediction[:, aspect_index]) == y_value[:, aspect_index]
    accuracy = np.mean(correct_predictions.astype(float))
    print("accuracy\t" + str(aspect_index) + "\t" + str(accuracy))

    mse = np.mean((all_prediction[:, aspect_index] - y_value[:, aspect_index]) ** 2)
    print("MSE\t" + str(aspect_index) + "\t" + str(mse))

    r2 = r2_score(y_value[:, aspect_index], all_prediction[:, aspect_index])
    print("R2 \t" + str(aspect_index) + "\t" + str(r2))
