from __future__ import division
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging
import readers
from  writer import MultiFileWriter, serialize_video_level_example
import utils
from scipy.linalg import eig
from joblib import Parallel, delayed


FLAGS = flags.FLAGS
# training data
flags.DEFINE_string("input_frame_file_pattern", "data/frame_level/train*.tfrecord",
                    "Input file pattern")
flags.DEFINE_string("feature_names", "rgb", "Name of the feature "
                    "to use for training.")
flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")
# output file name
flags.DEFINE_string("pooled_file_name", "data/frame_level/eigenvectors.tfrecord",
                    "Output filename containing covariance and Eigen decomposition.")

flags.DEFINE_string("sample_time_steps", "25,100", "Number of time steps for eigen pooling.")

flags.DEFINE_integer("num_readers", 2, "Number of threads for readers.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")

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
        tf.train.match_filenames_once(FLAGS.input_frame_file_pattern), num_epochs=1)
    reader = readers.YT8MFrameFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes)

    # training_data = [
    #     reader.prepare_reader(filename_queue) for _ in range(int(FLAGS.num_readers))
    # ]
    training_data = reader.prepare_reader(filename_queue)
    return training_data

def getListOutputFiles(prefix='', dir='', num_files=4096):
    from tempfile import mktemp
    files = set()
    count = 0
    while count < num_files:
        name = mktemp(prefix=prefix, dir=dir)
        if name not in files:
            files.add(name)
            count += 1
    files = sorted(list(files))
    return files

def main():
    logging.set_verbosity(tf.logging.INFO)
    construct_eigen_pooling_features()

def uniform_sampling(sample_nframes, video_matrix, num_frames):
    with tf.variable_scope("uniform_sampling"):
        # indexing
        sample_idxs = tf.range(sample_nframes, dtype=tf.float32)
        sample_freq = tf.cast(num_frames / sample_nframes, tf.float32)
        indices = tf.cast(tf.floor(sample_idxs * sample_freq), dtype=tf.int32)
        sampled_vid_feats = tf.gather(video_matrix, indices)
    return sampled_vid_feats

def batch_covariance(vid_feats, samp_freq):
    with tf.variable_scope("batch_covar_{}".format(samp_freq), reuse=True):
        perm_vid_feats = tf.transpose(vid_feats)
        vid_covar = tf.matmul(vid_feats, perm_vid_feats)
        red_vid_covar = tf.reduce_sum(vid_covar, axis=0)
        pooled_covar = tf.Variable(np.zeros((samp_freq, samp_freq)),
                                   dtype=tf.float32, trainable=False)
        accumulator = pooled_covar.assign(pooled_covar + red_vid_covar)
    return accumulator

def compute_global_mean(vid_feats, samp_freq):
    with tf.variable_scope("global_mean_{}".format(samp_freq)):
        global_mean = tf.Variable(np.zeros((samp_freq, 1024)),
                                  dtype=tf.float32, trainable=False)
        red_feats = tf.reduce_sum(vid_feats, axis=0)
        global_mean = global_mean.assign(global_mean + red_feats)
    return global_mean

def construct_eigen_pooling_features():
    feat_dim = 1024
    sampling_freqs = [int(t) for t in FLAGS.sample_time_steps.strip(',').split(',')]
    tot_num_frames_op = tf.Variable(0, dtype=tf.int32, trainable=False)

    training_data = parse_frame_level_features()
    video_ids, video_matrix, labels, num_frames = training_data
    logging.info("Using batch size of %d." % FLAGS.batch_size)
    tot_num_frames_op = tot_num_frames_op.assign(
        tot_num_frames_op + 1)
    video_matrix = tf.squeeze(tf.convert_to_tensor(video_matrix))
    num_frames = tf.convert_to_tensor(num_frames)

    # sampling
    frame_samplers = [
        uniform_sampling(nf, video_matrix, num_frames)
        for nf in sampling_freqs
    ]
    # create eigen pooling features for each sampling frequency
    pooled_covar_accs = [
        batch_covariance(samp, nf)
        for samp, nf in zip(frame_samplers, sampling_freqs)
    ]
    # compute the global mean
    global_means = [
        compute_global_mean(samp, nf)
        for samp, nf in zip(frame_samplers, sampling_freqs)
    ]

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
            # if nvids >= 5000:
                # break
            acc_covars, gb_means, nvids = sess.run(
                [pooled_covar_accs, global_means, tot_num_frames_op])
            # nvids = nvids + batch_size
            end = time.time() - start
            if printed + 10000 < nvids:
                printed = nvids
                logging.info("Processed %d in %.3f sec @ %.3f ms/ex" % (
                    nvids, end, (end*1000)/nvids))
    except tf.errors.OutOfRangeError:
        logging.info('Done iterating -- epoch limit reached')

    # calculate mean covariance matrix
    acc_covars = [cov/nvids for cov in acc_covars]
    gb_means = [mu/nvids for mu in gb_means]
    logging.info("Skipped: " + str(skipped))
    logging.info("Calculating the Eigendecomposition of the Covariance matrix")
    for cov, mu, nf in zip(acc_covars, gb_means, sampling_freqs):
        fname = os.path.splitext(FLAGS.pooled_file_name)
        fname = fname[0] + '_%d%s' % (nf, fname[-1])
        logging.info("Writing cov mat, eigenvalues and eigenvectors to %s" % (fname))
        writer = tf.python_io.TFRecordWriter(fname)
        [lambdas, vr] = eig(cov)
        example = tf.train.Example(features=tf.train.Features(feature={
            'time_steps': _int64_feature(nf),
            'eigen_vectors': _float_feature(np.real(vr)),
            'eigen_values': _float_feature(np.real(lambdas)),
            'global_mean': _float_feature(mu),
            'raw_covar': _float_feature(cov)
        }))
        writer.write(example.SerializeToString())
        writer.close()

    # When done, ask the threads to stop.
    coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    logging.info("Completed %d examples in %.3f sec." % (nvids, time.time() - start))

if __name__ == '__main__':
    main()
