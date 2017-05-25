# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides readers configured for different datasets."""

import tensorflow as tf
import numpy as np
import utils
from tensorflow import logging


def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be
      cast to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized

class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


class YT8MAggregatedFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """

  def __init__(self,
               num_classes=4716,
               feature_sizes=[1024],
               feature_names=["mean_inc3"]):
    """Construct a YT8MAggregatedFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """

    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names

  def prepare_reader(self, filename_queue, batch_size=1024):
    """Creates a single reader thread for pre-aggregated YouTube 8M Examples.

    Args:
      filename_queue: A tensorflow queue of filename locations.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read_up_to(filename_queue, batch_size)

    tf.add_to_collection("serialized_examples", serialized_examples)
    return self.prepare_serialized_examples(serialized_examples)

  def prepare_serialized_examples(self, serialized_examples):
    # set the mapping from the fields to data types in the proto
    num_features = len(self.feature_names)
    assert num_features > 0, "self.feature_names is empty!"
    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(self.feature_names), len(self.feature_sizes))

    feature_map = {"video_id": tf.FixedLenFeature([], tf.string),
                   "labels": tf.VarLenFeature(tf.int64)}
    for feature_index in range(num_features):
      feature_map[self.feature_names[feature_index]] = tf.FixedLenFeature(
          [self.feature_sizes[feature_index]], tf.float32)

    features = tf.parse_example(serialized_examples, features=feature_map)
    labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
    labels.set_shape([None, self.num_classes])
    concatenated_features = tf.concat([
        features[feature_name] for feature_name in self.feature_names], 1)

    return features["video_id"], concatenated_features, labels, tf.ones([tf.shape(serialized_examples)[0]])

class YT8MFrameFeatureReader(BaseReader):
  """Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  """

  def __init__(self,
               num_classes=4716,
               feature_sizes=[1024],
               feature_names=["inc3"],
               max_frames=300):
    """Construct a YT8MFrameFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
      max_frames: the maximum number of frames to process.
    """

    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names
    self.max_frames = max_frames

  def get_video_matrix(self,
                       features,
                       feature_size,
                       max_frames,
                       max_quantized_value,
                       min_quantized_value):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
      num_frames: number of frames in the sequence
    """
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = utils.Dequantize(decoded_features,
                                      max_quantized_value,
                                      min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix, num_frames

  def prepare_reader(self,
                     filename_queue,
                     max_quantized_value=2,
                     min_quantized_value=-2):
    """Creates a single reader thread for YouTube8M SequenceExamples.

    Args:
      filename_queue: A tensorflow queue of filename locations.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A tuple of video indexes, video features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    return self.prepare_serialized_examples(serialized_example,
        max_quantized_value, min_quantized_value)

  def prepare_serialized_examples(self, serialized_example,
      max_quantized_value=2, min_quantized_value=-2):

    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={"video_id": tf.FixedLenFeature(
            [], tf.string),
                          "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in self.feature_names
        })

    # read ground truth labels
    labels = (tf.cast(
        tf.sparse_to_dense(contexts["labels"].values, (self.num_classes,), 1,
            validate_indices=False),
        tf.bool))

    # loads (potentially) different types of features and concatenates them
    num_features = len(self.feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(self.feature_names), len(self.feature_sizes))

    num_frames = -1  # the number of frames in the video
    feature_matrices = [None] * num_features  # an array of different features
    for feature_index in range(num_features):
      feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
          features[self.feature_names[feature_index]],
          self.feature_sizes[feature_index],
          self.max_frames,
          max_quantized_value,
          min_quantized_value)
      if num_frames == -1:
        num_frames = num_frames_in_this_feature
      else:
        tf.assert_equal(num_frames, num_frames_in_this_feature)

      feature_matrices[feature_index] = feature_matrix

    # cap the number of frames at self.max_frames
    num_frames = tf.minimum(num_frames, self.max_frames)

    # concatenate different features
    video_matrix = tf.concat(feature_matrices, 1)

    # convert to batch format.
    # TODO: Do proper batch reads to remove the IO bottleneck.
    batch_video_ids = tf.expand_dims(contexts["video_id"], 0)
    batch_video_matrix = tf.expand_dims(video_matrix, 0)
    batch_labels = tf.expand_dims(labels, 0)
    batch_frames = tf.expand_dims(num_frames, 0)

    return batch_video_ids, batch_video_matrix, batch_labels, batch_frames


class YT8MFrameTransformFeatureReader(BaseReader):
  """Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  """

  def __init__(self,
               transform='avg',  # 'eigen', 'avg'
               sample_time_steps=0,
               eigen_vec_file_name='',
               num_classes=4716,
               top_k_eigen_feats=10,
               feature_sizes=[1024],
               feature_names=["inc3"],
               max_frames=300):
    """Construct a YT8MFrameEigPoolFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
      max_frames: the maximum number of frames to process.
    """

    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))

    assert sample_time_steps != 0, "Time steps 0."
    if transform == 'eigen':
      assert eigen_vec_file_name is not '', "Enter path to eigenvectors."
    self.transform_type = transform
    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names
    self.max_frames = max_frames
    self.sample_time_steps = sample_time_steps
    if transform == 'eigen':
      self.top_k_eigen_feats = top_k_eigen_feats
      self.eigenvecs_np = self.load_eigen_matrix(eigen_vec_file_name).T

  def get_video_matrix(self,
                       features,
                       feature_size,
                       max_frames,
                       max_quantized_value,
                       min_quantized_value):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
      num_frames: number of frames in the sequence
    """
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = utils.Dequantize(decoded_features,
                                      max_quantized_value,
                                      min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix, num_frames

  def load_eigen_matrix(self, fname):
    time_steps = self.sample_time_steps
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
    eigenvecs_ten = tf.reshape(transform_data['eigen_vectors'], (time_steps, time_steps))
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        eigenvecs = sess.run(eigenvecs_ten)
        coord.request_stop()
        coord.join(threads)
    eigenvecs = eigenvecs[:, :self.top_k_eigen_feats]  # the top k eigen vecs
    return eigenvecs

  def prepare_reader(self,
                     filename_queue,
                     max_quantized_value=2,
                     min_quantized_value=-2):
    """Creates a single reader thread for YouTube8M SequenceExamples.

    Args:
      filename_queue: A tensorflow queue of filename locations.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A tuple of video indexes, video features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    return self.prepare_serialized_examples(serialized_example,
        max_quantized_value, min_quantized_value)

  def prepare_serialized_examples(self, serialized_example,
      max_quantized_value=2, min_quantized_value=-2):

    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={"video_id": tf.FixedLenFeature(
            [], tf.string),
                          "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in self.feature_names
        })

    # read ground truth labels
    labels = (tf.cast(
        tf.sparse_to_dense(contexts["labels"].values, (self.num_classes,), 1,
            validate_indices=False),
        tf.bool))

    # loads (potentially) different types of features and concatenates them
    num_features = len(self.feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(self.feature_names), len(self.feature_sizes))

    num_frames = -1  # the number of frames in the video
    feature_matrices = [None] * num_features  # an array of different features
    for feature_index in range(num_features):
      feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
          features[self.feature_names[feature_index]],
          self.feature_sizes[feature_index],
          self.max_frames,
          max_quantized_value,
          min_quantized_value)
      if num_frames == -1:
        num_frames = num_frames_in_this_feature
      else:
        tf.assert_equal(num_frames, num_frames_in_this_feature)

      if "rgb" in self.feature_names[feature_index]:
        # compute avg/eigen pooling on sampled rgb features
        samp_feature_matrix = self.uniform_sampling(
          feature_matrix, num_frames_in_this_feature)
        if self.transform_type == 'eigen':
          feature_matrix = self.apply_eigen_pooling(samp_feature_matrix)
        elif self.transform_type == 'avg':
          feature_matrix = self.compute_mean(samp_feature_matrix)
        elif self.transform_type == 'all':
          feature_matrix = samp_feature_matrix
      else:
        # compute video level avg features for audio
        feature_matrix = self.compute_mean(feature_matrix)
      if self.transform_type != 'all':
        feature_matrix = tf.reshape(feature_matrix, [-1])
      feature_matrices[feature_index] = feature_matrix

    # cap the number of frames at self.max_frames
    num_frames = tf.minimum(num_frames, self.max_frames)

    if self.transform_type == 'all':
      batch_video_matrix = feature_matrices
    else:
      # concatenate different features
      video_matrix = tf.concat(feature_matrices, 0)
      batch_video_matrix = tf.expand_dims(video_matrix, 0)

    # convert to batch format.
    # TODO: Do proper batch reads to remove the IO bottleneck.
    batch_video_ids = tf.expand_dims(contexts["video_id"], 0)
    batch_labels = tf.expand_dims(labels, 0)
    batch_frames = tf.expand_dims(num_frames, 0)

    return batch_video_ids, batch_video_matrix, batch_labels, batch_frames

  def uniform_sampling(self, video_mat, num_frames):
    with tf.variable_scope("uniform_sampling"):
      time_steps = self.sample_time_steps
      # indexing
      sample_idxs = tf.range(time_steps, dtype=tf.float32)
      sample_freq = tf.cast(num_frames / time_steps, tf.float32)
      indices = tf.cast(tf.floor(sample_idxs * sample_freq), dtype=tf.int32)
      # slice
      samp_video_mat = tf.gather(video_mat, indices)
    return samp_video_mat

  def apply_eigen_pooling(self, samp_video_mat):
    with tf.variable_scope("eigen_pooling"):
      eigen_vecs = tf.constant(self.eigenvecs_np, dtype=tf.float32)
      tr_video_mat = tf.matmul(eigen_vecs, samp_video_mat)
    return tr_video_mat

  def compute_mean(self, samp_video_mat):
    with tf.variable_scope("mean"):
      mu_video_mat = tf.reduce_mean(samp_video_mat, axis=0)
    return mu_video_mat
