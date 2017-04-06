from __future__ import division
import os
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging
from tempfile import mktemp
import utils
from Queue import Queue

logging.set_verbosity(tf.logging.INFO)

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

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_video_level_example(vid, features, label):
    label = np.where(label==True)[1]
    ex = tf.train.Example(features=tf.train.Features(feature={
        'video_id': _bytes_feature(vid.tostring()),
        'labels': _int64_feature(label),
        'mean_rgb': _float_feature(features)
    }))
    return ex.SerializeToString()

class MultiFileWriter(object):
    def __init__(self, num_files, num_ex_per_file, prefix, dir=''):
        self.num_files = num_files
        self.num_ex_per_file = num_ex_per_file
        self.num_ex_curr_file = 0
        self.file_names_q = self.getListOutputFiles(prefix, dir, num_files)
        name = self.file_names_q.get()
        self.writer = tf.python_io.TFRecordWriter(name)
        self.curr_file_name = name
        logging.info("Begin writing in %s" % name)

    def getListOutputFiles(self, prefix, dir, num_files):
        files = set()
        count = 0
        while count < num_files:
            name = mktemp(prefix=prefix, suffix='.tfrecord', dir=dir)
            if name not in files:
                files.add(name)
                count += 1
        files = sorted(list(files))
        q = Queue(maxsize=len(files))
        for name in files:
            q.put(name)
        return q

    def writer_init(self):
        if self.num_ex_curr_file < self.num_ex_per_file:
            return
        self.num_ex_curr_file = 0
        self.writer.close()
        next_name = self.file_names_q.get()
        self.writer = tf.python_io.TFRecordWriter(next_name)
        self.curr_file_name = next_name
        logging.info("Begin writing in %s" % next_name)

    def write(self, serialized_examples):
        if not isinstance(serialized_examples, list):
            serialized_examples = [serialized_examples]
        for ex in serialized_examples:
            self.writer_init()
            self.num_ex_curr_file += 1
            self.writer.write(ex)

    def close(self):
        self.writer.close()
        if self.num_ex_curr_file == 0:
            os.unlink(self.curr_file_name)
        logging.info("Closing writer.")
