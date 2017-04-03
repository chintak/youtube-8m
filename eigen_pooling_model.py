from __future__ import division
import time
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

def mapped(fn, arrs):
    ii = tf.range(0, tf.shape(arrs[0])[0])
    out = tf.map_fn(lambda i: fn(*[arr[i] for arr in arrs]), ii, dtype=tf.float32)
    return out

def parse_frame_level_features():
    # Frame level features
    # feature_names, feature_sizes = "rgb", "1024"
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)

    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(FLAGS.input_file_pattern), num_epochs=1)
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

    # Eigen pooling
    training_data = parse_frame_level_features()
    video_ids, video_matrix, labels, num_frames = map(list, zip(*training_data))
    batch_size = len(video_ids)
    video_matrix = tf.squeeze(tf.convert_to_tensor(video_matrix))
    num_frames = tf.convert_to_tensor(num_frames)
    sample_idxs = tf.constant(np.tile(np.arange(sample_nframes), (batch_size,1)),
                              dtype=tf.float32)

    sample_freq = tf.cast(num_frames / sample_nframes, tf.float32)
    # indexing
    b = tf.reshape(tf.range(0, batch_size), (batch_size, 1))
    b = tf.tile(b, (1, sample_nframes))
    pick = tf.cast(tf.floor(sample_idxs * sample_freq), dtype=tf.int32)
    indices = tf.stack([b, pick], axis=2)

    sampled_vid_feats = tf.gather_nd(video_matrix, indices)
    perm_sampled_vid_feats = tf.transpose(sampled_vid_feats, perm=[0, 2, 1])
    vid_covar = mapped(lambda a,b: tf.matmul(a, b),
                       [sampled_vid_feats, perm_sampled_vid_feats])
    vid_covar = tf.reshape(vid_covar, (batch_size, sample_nframes, sample_nframes))
    red_vid_covar = tf.reduce_sum(vid_covar, axis=0)
    pooled_covar = tf.Variable(np.zeros((sample_nframes, sample_nframes)),
                               dtype=tf.float32, trainable=False)
    accumulator = pooled_covar.assign(pooled_covar + red_vid_covar)

    # Create tensorflow Session
    sess = tf.Session()

    # Initialize the variables (like the epoch counter).
    tf.global_variables_initializer().run(session=sess)
    tf.local_variables_initializer().run(session=sess)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        min_nframes = 1e3
        nvids = 0
        skipped = 0
        printed = 0
        start = time.time()
        while not coord.should_stop():
            # Run training steps or whatever
            # if nvids >= 1000:
                # break
            acc_covars = sess.run(accumulator)
            nvids = nvids + batch_size
            end = time.time() - start
            if printed + 10000 < nvids:
                printed = nvids
                logging.info("Processed %d in %.3f sec @ %.3f ms" % (
                    nvids, end, (end*1000)/nvids))
        # calculate mean covariance matrix
        acc_covars /= nvids
        logging.info("Skipped: " + str(skipped))
        logging.info("Calculating the Eigendecomposition of the Covariance matrix")
        [lambdas, vr] = eig(acc_covars)
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
        logging.info('Done iterating -- epoch limit reached')

    # When done, ask the threads to stop.
    coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    logging.info("Completed %d examples." % nvids)

if __name__ == '__main__':
    main()
