#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import shutil
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
from tensorflow.python.platform import gfile
from tensorflow.contrib import lookup
from pathlib import Path
from tensorflow.python import debug as tf_debug
from tensorflow.contrib import rnn
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV


tf.logging.set_verbosity(tf.logging.INFO)

# variables set by init()
TRAIN_STEPS = 5000
N_WORDS = -1

# hardcoded into graph
BATCH_SIZE = 64

# describe your data
TARGETS = ["EAP", "HPL", "MWS"]
MAX_DOCUMENT_LENGTH = 100
CSV_COLUMNS = ["id","text","author"]
CSV_COLUMNS_TEST = ["id","text"]
LABEL_COLUMN = 'author'
DEFAULTS = [['null'], ['null'], ['null']]
DEFAULTS_TEST = [['null'], ['null']]
PADWORD = 'ZYXW'
DATA_DIR = Path("/media/ssd_data/kagle/spooky-author-identification/input")
TRAIN_DATA = DATA_DIR / Path("train.csv")
VAL_DATA = DATA_DIR / Path("val.csv")
TEST_DATA = DATA_DIR / Path("test.csv")
WORD_VOCAB_FILE = DATA_DIR / Path("vocab_words")

def init(num_steps):
  global TRAIN_STEPS, N_WORDS
  TRAIN_STEPS = num_steps
  N_WORDS = save_vocab(str(TRAIN_DATA), 'text', str(WORD_VOCAB_FILE));

def save_vocab(trainfile, txtcolname, outfilename):
  if trainfile.startswith('gs://'):
    import subprocess
    tmpfile = "vocab.csv"
    subprocess.check_call("gsutil cp {} {}".format(trainfile, tmpfile).split(" "))
    filename = tmpfile
  else:
    filename = trainfile
  #df = pd.read_csv(filename, header=None, sep='\t', names=['source', 'title'])
  df = pd.read_csv(filename, encoding="utf8")
  # the text to be classified
  vocab_processor = tflearn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=1)
  vocab_processor.fit(df[txtcolname])

  with gfile.Open(outfilename, 'wb') as f:
    f.write("{}\n".format(PADWORD))
    for word, index in vocab_processor.vocabulary_._mapping.items():
      f.write("{}\n".format(word))


  nwords = len(vocab_processor.vocabulary_)
  print('{} words into {}'.format(nwords, outfilename))
  return nwords + 2  # PADWORD and <UNK>

def read_dataset(prefix):
  # use prefix to create filename
  if prefix == 'train':
    mode = tf.contrib.learn.ModeKeys.TRAIN
    filename = str(TRAIN_DATA)
  else:
    mode = tf.contrib.learn.ModeKeys.EVAL
    #TODO need validate dataset
    filename = str(VAL_DATA)

  # the actual input function passed to TensorFlow
  def _input_fn():
    # could be a path to one file or a file pattern.
    input_file_names = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(input_file_names, shuffle=True)
 
    # read CSV
    reader = tf.TextLineReader(skip_header_lines=1)
    _, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)
    #value = tf.train.shuffle_batch([value], BATCH_SIZE, capacity=10*BATCH_SIZE, min_after_dequeue=BATCH_SIZE, enqueue_many=True, allow_smaller_final_batch=False)
    value_column = tf.expand_dims(value, -1)
    columns = tf.decode_csv(value_column, record_defaults=DEFAULTS, field_delim=',')
    features = dict(zip(CSV_COLUMNS, columns))
    label = features.pop(LABEL_COLUMN)

    # make targets numeric
    table = tf.contrib.lookup.index_table_from_tensor(
                   mapping=tf.constant(TARGETS), num_oov_buckets=0, default_value=-1)
    target = table.lookup(label)

    return features, target
  
  return _input_fn





def dense(x, size, scope):
    return tf.contrib.layers.fully_connected(x, size,
                                             activation_fn=None,
                                             scope=scope)


def dense_batch_relu(x, phase, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, 200,
                                               activation_fn=None,
                                               scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1,
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn')
        return tf.nn.relu(h2, 'relu')

# CNN model parameters
EMBEDDING_SIZE = 70
WINDOW_SIZE = EMBEDDING_SIZE//2
STRIDE = int(WINDOW_SIZE/2)

hidden_layer_size = 64
#_seqlens = MAX_DOCUMENT_LENGTH

def rnn_model(features, target, mode):
    table = lookup.index_table_from_file(vocabulary_file=str(WORD_VOCAB_FILE), num_oov_buckets=1, default_value=-1)

    print('features={}'.format(features))  # (?, 20)

    def my_func(x, target):
        # x will be a numpy array with the contents of the placeholder below
        for _x in zip(x,target):
            print(_x)
        return x

    #f = tf.py_func(my_func, [features["text"], target], tf.string)
    # string operations
    #titles = tf.squeeze(features['text'], [1])
    #titles = tf.squeeze(f, [1])
    # string operations
    #words = tf.string_split(titles)
    words = tf.string_split(features["text"])
    #TODO: calc sequence_length
    #words = tf.Print(words, [words])
    densewords = tf.sparse_tensor_to_dense(words, default_value=PADWORD)
    numbers = table.lookup(densewords)
    padding = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
    padded = tf.pad(numbers, padding)
    sliced = tf.slice(padded, [0,0], [-1, MAX_DOCUMENT_LENGTH])
    print('words_sliced={}'.format(words))  # (?, 20)

    # layer to take the words and convert them into vectors (embeddings)
    print(N_WORDS)
    embeds = tf.contrib.layers.embed_sequence(sliced, vocab_size=N_WORDS, embed_dim=EMBEDDING_SIZE)
    print('words_embed={}'.format(embeds)) # (?, 20, 10)

    n_classes = len(TARGETS)
    print(n_classes, TARGETS)

    with tf.variable_scope("lstm"):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)

        outputs, states = tf.nn.dynamic_rnn(lstm_cell, embeds, dtype=tf.float32)

    last = outputs[:,-1,:]#tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

    # ===========

    fc1bn = dense_batch_relu(last, (mode == "train"), "dense1")

    fc1bn_do = tf.contrib.layers.dropout(fc1bn, keep_prob=0.8)



    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    ##cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)
    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each
    # unit.
    ##_, encoding = tf.nn.rnn(cell, word_list, dtype=tf.float32)

    logits = tf.contrib.layers.fully_connected(fc1bn_do, n_classes, activation_fn=None)
    print('logits={}'.format(logits)) # (?, 3)
    predictions_dict = {
        'author': tf.gather(TARGETS, tf.argmax(logits, 1)),
        'class': tf.argmax(logits, 1),
        'prob': tf.nn.softmax(logits)
    }

    if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(target, logits)
        train_op = tf.contrib.layers.optimize_loss(
            loss,
            tf.contrib.framework.get_global_step(),
            optimizer='Adam',
            #optimizer='SGD',
            learning_rate=0.001)
    else:
        loss = None
        train_op = None

    return tflearn.ModelFnOps(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op)


def cnn_model(features, target, mode):
    table = lookup.index_table_from_file(vocabulary_file=str(WORD_VOCAB_FILE), num_oov_buckets=1, default_value=-1)

    def my_func(x, target):
        # x will be a numpy array with the contents of the placeholder below
        for _x in zip(x,target):
            print(_x)
        return x
    f = tf.py_func(my_func, [features["text"], target], tf.string)
    # string operations
    titles = tf.squeeze(features['text'], [1])
    #titles = tf.squeeze(f, [1])
    #features['text']
    words = tf.string_split(titles)
    #words = tf.Print(words, [words])
    densewords = tf.sparse_tensor_to_dense(words, default_value=PADWORD)
    numbers = table.lookup(densewords)
    padding = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
    padded = tf.pad(numbers, padding)
    sliced = tf.slice(padded, [0,0], [-1, MAX_DOCUMENT_LENGTH])
    print('words_sliced={}'.format(words))  # (?, 20)

    # layer to take the words and convert them into vectors (embeddings)
    print(N_WORDS)
    embeds = tf.contrib.layers.embed_sequence(sliced, vocab_size=N_WORDS, embed_dim=EMBEDDING_SIZE)
    print('words_embed={}'.format(embeds)) # (?, 20, 10)
    
    # now do convolution
    with tf.name_scope("convolution"):
        conv = tf.contrib.layers.conv2d(embeds, 1, WINDOW_SIZE, stride=STRIDE, padding='SAME') # (?, 4, 1)
        conv = tf.nn.relu(conv) # (?, 4, 1)
        words = tf.squeeze(conv, [2]) # (?, 4)
        print('words_conv={}'.format(words)) # (?, 4)


    n_classes = len(TARGETS)
    print(n_classes, TARGETS)

    fc1bn = dense_batch_relu(words, (mode == tf.contrib.learn.ModeKeys.TRAIN), "dense1")

    fc1bn_do = tf.contrib.layers.dropout(fc1bn, keep_prob=0.9)

    logits = tf.contrib.layers.fully_connected(fc1bn_do, n_classes, activation_fn=None)
    print('logits={}'.format(logits)) # (?, 3)
    predictions_dict = {
      'author': tf.gather(TARGETS, tf.argmax(logits, 1)),
      'class': tf.argmax(logits, 1),
      'prob': tf.nn.softmax(logits)
    }

    if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
       loss = tf.losses.sparse_softmax_cross_entropy(target, logits)
       train_op = tf.contrib.layers.optimize_loss(
         loss,
         tf.contrib.framework.get_global_step(),
         optimizer='Adam',
         #optimizer='SGD',
         learning_rate=0.001)
    else:
       loss = None
       train_op = None

    return tflearn.ModelFnOps(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op)


def serving_input_fn():
    feature_placeholders = {
      'text': tf.placeholder(tf.string, [None]),
    }
    #key: tf.expand_dims(tensor, -1)
    features = {
      key: tensor
      for key, tensor in feature_placeholders.items()
    }
    return tflearn.utils.input_fn_utils.InputFnOps(
      features,
      None,
      feature_placeholders)

def get_train():
  return read_dataset('train')

def get_valid():
  return read_dataset('eval')

from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.python import debug as tf_debug

hooks = [tf_debug.LocalCLIDebugHook()]


#============================================================


training_set = pd.read_csv(str(TRAIN_DATA), encoding="utf8")
test_set = pd.read_csv(str(VAL_DATA), encoding="utf8")
prediction_set = pd.read_csv(str(TEST_DATA), encoding="utf8")
#vectorizer = DV(sparse=False)
#vectorizer.fit(training_set[LABEL_COLUMN])

def get_input_fn(data_set, num_epochs=None, shuffle=True, infer=False):
    # make targets numeric
    target = None
    if not infer:
        target, unique = pd.factorize(data_set[LABEL_COLUMN], sort=True)
        print(unique)
        target = pd.Series(target)
    return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in CSV_COLUMNS_TEST}),
      y=target,
      num_epochs=num_epochs,
      shuffle=shuffle)


#============================================================

def experiment_fn(output_dir):
    # run experiment
    return tflearn.Experiment(
        tflearn.Estimator(model_fn=rnn_model, model_dir=output_dir),
        #train_input_fn=get_train(),
        train_input_fn=get_input_fn(training_set),
        #eval_input_fn=get_valid(),
        eval_input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False),
        eval_steps=50,
        eval_metrics={
            'acc': tflearn.MetricSpec(
                metric_fn=metrics.streaming_accuracy, prediction_key='class'
            )
        },
        export_strategies=[saved_model_export_utils.make_export_strategy(
            serving_input_fn,
            default_output_alternative_key=None,
            exports_to_keep=1
        )],
        train_steps = TRAIN_STEPS,
        #train_monitors=hooks,
        #eval_hooks=hooks
    )




def get_estimator(run_config, params):
    """Return the model as a Tensorflow Estimator object.
    Args:
         run_config (RunConfig): Configuration for Estimator run.
         params (HParams): hyperparameters.
    """
    return tf.estimator.Estimator(
        model_fn=rnn_model,  # First-class function
        params=params,  # HParams
        config=run_config  # RunConfig
    )

def infer(output_dir):
    estimator = tflearn.Estimator(model_fn=rnn_model, model_dir=output_dir)
    result = estimator.predict(input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False, infer=True))
    probs = [x["prob"] for x in result]
    df = pd.DataFrame(probs, columns=TARGETS)
    df = pd.concat([prediction_set, df], axis=1)
    print(df.head())
    COL_TO_SAVE = ["id"] + TARGETS
    df[COL_TO_SAVE].to_csv("out.csv", index=False)
