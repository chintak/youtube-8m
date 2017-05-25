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
from contextlib import contextmanager
import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import starmap
from multiprocessing import Pool
from functools import reduce
from operator import add


FLAGS = flags.FLAGS
# training data
flags.DEFINE_string("input_fname_files", "data/frame_level/files_train.txt",
                    "Input file pattern")
flags.DEFINE_string("output_fname_files", "data/eigen_level/files_train.txt",
                    "Input file pattern")
flags.DEFINE_string("feature_names", "rgb, audio", "Name of the feature "
                    "to use for training.")
flags.DEFINE_string("feature_sizes", "1024, 128", "Length of the feature vectors.")

flags.DEFINE_string("eigen_vec_file_name", "",
                    "Path to tfrecord file containing eigen vectors.")
flags.DEFINE_integer("eigen_k_vecs", 3,
                     "Number of eigen vectors to use for pooling.")
flags.DEFINE_integer("transform_time_steps", 25,
                     "Number of time steps for eigen pooling.")

flags.DEFINE_integer("max_w", 16, "Number of threads for readers.")

def _int64_feature(value_list):
    if isinstance(value_list, np.ndarray):
        value_list = value_list.flatten().tolist()
    elif isinstance(value_list, float):
        value_list = [value_list]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))

def _float_feature(value_list):
    if isinstance(value_list, np.ndarray):
        value_list = value_list.flatten().tolist()
    elif isinstance(value_list, float):
        value_list = [value_list]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))

def _bytes_feature(value_list):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))

def serialize_video_level_example(eig_fts, rank_fts, mu_fts, max_fts, audio, vid, label):
    features={}
    for i in range(eig_fts.shape[0]):
        features['eig%d' % (i+1)] = _float_feature(np.ravel(eig_fts[i, :]))
    features['rank'] = _float_feature(rank_fts)
    features['mean'] = _float_feature(mu_fts)
    features['max'] = _float_feature(max_fts)
    features['mean_audio'] = _float_feature(audio)
    features['video_id'] = _bytes_feature(vid)
    label = np.where(label==True)[1]
    features['labels'] = _int64_feature(label)
    ex = tf.train.Example(features=tf.train.Features(feature=features))
    return ex.SerializeToString()

def parse_frame_level_features(sess, input_file_name):
    # Frame level features
    # feature_names, feature_sizes = "rgb", "1024"
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)

    reader = readers.YT8MFrameTransformFeatureReader(
        transform='all',
        sample_time_steps=FLAGS.transform_time_steps,
        eigen_vec_file_name=FLAGS.eigen_vec_file_name,
        top_k_eigen_feats=FLAGS.eigen_k_vecs,
        feature_names=feature_names, feature_sizes=feature_sizes)

    filename_queue = tf.train.string_input_producer([input_file_name])
    training_data = reader.prepare_reader(filename_queue)
    return training_data

def load_eigen_matrix(fname, time_steps, topk):
    filename_queue = tf.train.string_input_producer([fname])
    reader = tf.TFRecordReader()
    _, serialized_feats = reader.read(filename_queue)
    transform_data = tf.parse_single_example(
        serialized_feats,
        features={
            'time_steps': tf.FixedLenFeature([], tf.int64),
            'eigen_vectors': tf.FixedLenFeature([time_steps**2], tf.float32),
#            'eigen_values': tf.FixedLenFeature([time_steps], tf.float32),
#            'global_mean': tf.FixedLenFeature([], tf.float32),
#            'raw_covar': tf.FixedLenFeature([time_steps**2], tf.float32)
        })
    eigenvecs_ten = tf.reshape(
        transform_data['eigen_vectors'], (time_steps, time_steps))
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        eigenvecs = sess.run(eigenvecs_ten)
        coord.request_stop()
        coord.join(threads)
    eigenvecs = eigenvecs[:, :topk]  # the top k eigen vecs
    with tf.variable_scope("eigen_vecs"):
        eigenvecs_tf = tf.constant(eigenvecs.T, tf.float32)
    return eigenvecs_tf

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

def apply_eigen_pooling(samp_video_mat, evecs):
    with tf.variable_scope("eigen_pooling"):
      tr_video_mat = tf.matmul(evecs, samp_video_mat)
    return tr_video_mat

def compute_mean(samp_video_mat):
    with tf.variable_scope("mean"):
      mu_video_mat = tf.reduce_mean(samp_video_mat, axis=0)
    return mu_video_mat

def transform_eig_max_avg_rank(file, out=None):
    if out is None:
        input_file, output_file = file[0], file[1]
    else:
        input_file, output_file = file, out
# def transform_eig_max_avg_rank(input_file, output_file):
    feat_dim = 1024
    time_steps = FLAGS.transform_time_steps
    # Eigen pooling
    eigenvecs = load_eigen_matrix(
        FLAGS.eigen_vec_file_name, FLAGS.transform_time_steps, FLAGS.eigen_k_vecs)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    training_data = parse_frame_level_features(sess, input_file)
    video_ids, feature_matrix, labels, num_frames = training_data
    if len(feature_matrix) == 2:
        video_matrix = feature_matrix[0]
        audio_matrix = feature_matrix[1]
    num_frames = tf.convert_to_tensor(num_frames)

    with tf.variable_scope("eigen_pool"):
      eig_video_mat = tf.matmul(eigenvecs, video_matrix)
    with tf.variable_scope("mean_pool"):
      mu_video_mat = tf.reduce_mean(video_matrix, axis=0)
    with tf.variable_scope("max_pool"):
      max_video_mat = tf.reduce_max(video_matrix, axis=0)
    with tf.variable_scope("rank"):
      # weights = tf.constant(
      #     [-0.786015, -0.402709, -0.227027, -0.120554, -0.048684, 0.002424, 0.039690,
      #      0.067068, 0.087032, 0.101229, 0.110811, 0.116619, 0.119281, 0.119281,
      #      0.116999, 0.112740, 0.106751, 0.099236, 0.090363, 0.080276, 0.069096,
      #      0.056928, 0.043860, 0.029972, 0.015332],
      #     dtype=tf.float32, shape=[1, time_steps])
        weights = tf.constant(
            [-0.590829, -0.410256, -0.321794, -0.264035, -0.221628, -0.188431,
             -0.161376, -0.138707, -0.119327, -0.102506, -0.087732, -0.074632,
             -0.062928, -0.052406, -0.042895, -0.034261, -0.026396, -0.019207,
             -0.012620, -0.006573, -0.001009, 0.004115, 0.008841, 0.013202,
             0.017230, 0.020951, 0.024389, 0.027564, 0.030495, 0.033200,
             0.035692, 0.037987, 0.040096, 0.042031, 0.043801, 0.045416,
             0.046886, 0.048217, 0.049417, 0.050492, 0.051450, 0.052295,
             0.053034, 0.053670, 0.054209, 0.054655, 0.055011, 0.055283,
             0.055473, 0.055585, 0.055621, 0.055585, 0.055480, 0.055308,
             0.055072, 0.054773, 0.054415, 0.053999, 0.053527, 0.053002,
             0.052424, 0.051796, 0.051120, 0.050396, 0.049626, 0.048813,
             0.047956, 0.047057, 0.046119, 0.045141, 0.044124, 0.043071,
             0.041982, 0.040857, 0.039699, 0.038507, 0.037283, 0.036028,
             0.034742, 0.033426, 0.032080, 0.030707, 0.029306, 0.027877,
             0.026422, 0.024942, 0.023436, 0.021905, 0.020351, 0.018773,
             0.017172, 0.015548, 0.013903, 0.012236, 0.010547, 0.008839,
             0.007110, 0.005361, 0.003593, 0.001806],
            dtype=tf.float32, shape=[1, time_steps])
        rank_video_mat = tf.matmul(weights, video_matrix)

    # Setup TFRecord writer
    writer = tf.python_io.TFRecordWriter(output_file)
    # logging.info('Begin writing %s file.' % (output_file))

    # Initialize the variables (like the epoch counter).
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    # Start input enqueue threads.
    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            nvids = 0
            printed = 0
            start = time.time()
            start_vid = ''
            while not coord.should_stop():
                # Run training steps or whatever
                rank_ft, mu_ft, max_ft, eig_ft, aud, vid_id, vid_lab = sess.run(
                    [rank_video_mat, mu_video_mat, max_video_mat,
                     eig_video_mat, audio_matrix, video_ids, labels])
                if vid_id[0] == start_vid:
                    break
                if nvids == 0:
                    start_vid = vid_id[0]
                nvids = nvids + 1
                ser_examples = serialize_video_level_example(
                    eig_ft, rank_ft, mu_ft, max_ft, aud, vid_id, vid_lab)
                writer.write(ser_examples)
                # end = time.time() - start
                # if printed + 10000 < nvids:
                #     printed = nvids
                #     logging.info("Processed %d from %s in %.3f sec @ %.3f ms/ex" % (
                #         nvids, input_file, end, (end*1000)/nvids))

        except KeyboardInterrupt:
            logging.info('Interrupted.')
        except tf.errors.OutOfRangeError:
            logging.info('Done iterating -- epoch limit reached')

        writer.close()
        # logging.info("Closing file %s." % (output_file))
        # When done, ask the threads to stop.
        coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        logging.info(
            "Wrote %d examples to %s in %.3f sec." % (
                nvids, output_file, time.time() - start))

def read_file_names(fname):
    with open(fname, 'r') as fp:
        names = fp.read().strip().split('\n')
    return names

@contextmanager
def terminating(thing):
    try:
        yield thing
    finally:
        thing.terminate()

def main_mp():
    logging.set_verbosity(tf.logging.INFO)
    input_files = read_file_names(FLAGS.input_fname_files)
    output_files = read_file_names(FLAGS.output_fname_files)
    with terminating(mp.Pool(processes=FLAGS.max_w)) as pool:
        try:
            res = pool.imap(transform_eig_max_avg_rank,
                            [(inp, out) for inp, out in zip(input_files, output_files)],
                            chunksize=512)
            print "Pool:", [i for i in res]
        except KeyboardInterrupt:
            pass

def main_futures():
    logging.set_verbosity(tf.logging.INFO)
    input_files = read_file_names(FLAGS.input_fname_files)
    output_files = read_file_names(FLAGS.output_fname_files)
    try:
        with ProcessPoolExecutor(max_workers=FLAGS.max_w) as executor:
            futures = {executor.submit(transform_eig_max_avg_rank, inp, out)
                       for inp, out in zip(input_files, output_files)}
            for ft in futures:
                print ft.result()
            concurrent.futures.wait(futures)
    except KeyboardInterrupt:
        for ft in futures.values():
            ft.shutdown()
        concurrent.futures.wait(futures)

def main_futures_map():
    logging.set_verbosity(tf.logging.INFO)
    input_files = read_file_names(FLAGS.input_fname_files)
    output_files = read_file_names(FLAGS.output_fname_files)
    files = [[inp, out] for inp, out in zip(input_files, output_files)]
    with ProcessPoolExecutor(max_workers=FLAGS.max_w) as executor:
        try:
            for res in executor.map(transform_eig_max_avg_rank, files, chunksize=1024):
                print res
        except KeyboardInterrupt:
            executor.cancel()

if __name__ == '__main__':
    main_futures_map()
