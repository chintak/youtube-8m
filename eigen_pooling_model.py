from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging
import readers
import utils
from scipy.linalg import eig

FLAGS = flags.FLAGS
# training data
flags.DEFINE_string("input_file_pattern", "data/frame_level/train*.tfrecord",
                    "Input file pattern")
flags.DEFINE_string("feature_names", "rgb", "Name of the feature "
                    "to use for training.")
flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")
# output file name
flags.DEFINE_string("output_file_name", "data/frame_level/eigenvectors.tfrecord",
                    "Output filename containing covariance and Eigen decomposition.")

flags.DEFINE_string("num_readers", 2, "Number of threads for readers.")

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value_list):
    if isinstance(value_list, np.ndarray):
        value_list = value_list.flatten().tolist()
    elif isinstance(value_list, float):
        value_list = [value_list]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parse_frame_level_features():
    # Frame level features
    # feature_names, feature_sizes = "rgb", "1024"
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)

    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(FLAGS.input_file_pattern))
    reader = readers.YT8MFrameFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes)

    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(int(FLAGS.num_readers))
    ]
    return training_data

def parse_eigen_vectors():
    tf.parse_single_example()

def main():
    logging.set_verbosity(tf.logging.INFO)
    feat_dim = 1024
    sample_nframes = 100
    sample_idx = np.arange(sample_nframes)
    training_data = parse_frame_level_features()
    sess = tf.Session()

    init_op = tf.global_variables_initializer()
    # Initialize the variables (like the epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        min_nframes = 1e3
        nvids = 0
        skipped = 0
        acc_covar = np.zeros((sample_nframes, sample_nframes), dtype=np.float32)
        while not coord.should_stop():
            # Run training steps or whatever
            if nvids >= 1000:
                break
            examples = sess.run(training_data)
            for v_id, v_feat, v_labels, nframes in examples:
                nframes = nframes[0]
                sample_freq = nframes // sample_nframes
                v_feat = np.squeeze(v_feat)
                idx = np.floor(sample_idx * sample_freq).astype(np.uint16)
                samp_feat = v_feat[idx, :]
                if samp_feat.shape[0] < sample_nframes:
                    skipped = skipped + 1
                    continue
                acc_covar = acc_covar + np.matmul(samp_feat, samp_feat.T)
                nvids = nvids + 1
                min_nframes = min(min_nframes, nframes)
                if nvids % 10 == 0:
                    logging.info("Processed %d" % nvids)
        # calculate mean covariance matrix
        acc_covar /= nvids
        logging.info("Skipped: " + str(skipped))
        logging.info("Calculating the Eigendecomposition of the Covariance matrix")
        [lambdas, vr] = eig(acc_covar)
        logging.info("Writing covariance matrix, eigen values and eigen vectors to %s" % (
            FLAGS.output_file_name))
        writer = tf.python_io.TFRecordWriter(FLAGS.output_file_name)
        example = tf.train.Example(features=tf.train.Features(feature={
            'time_steps': _int64_feature(sample_nframes),
            'eigen_vectors': _float_feature(vr),
            'eigen_values': _float_feature(np.real(lambdas))
        }))
        writer.write(example.SerializeToString())

    except tf.errors.OutOfRangeError:
        logging.error('Done iterating -- epoch limit reached')

    # When done, ask the threads to stop.
    coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    logging.info("Complete.")

if __name__ == '__main__':
    main()
