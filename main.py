"""
  Deep Learning for Janken Game using MLP
  main.py
  2017.12  Takata
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import datetime as dt
import time
from janken import Janken

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer( 'test_data', 10, 'number of test data')
tf.app.flags.DEFINE_integer( 'training_data', 1000, 'number of training data')

def main( argv=None ):
  janken = Janken()
  training_data = janken.get_training_data( FLAGS.training_data )
  test_data = janken.get_training_data( FLAGS.test_data )

if __name__ == '__main__':
  tf.app.run()

