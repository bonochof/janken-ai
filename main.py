"""
  Deep Learning for Janken Game using MLP
  main.py
  2017.12  Takata
"""

from janken import Janken
from train import *

def main( argv=None ):
  janken = Janken()
  training_data = janken.get_training_data( FLAGS.training_data )
  test_data = janken.get_training_data( FLAGS.test_data )

  train_and_test( training_data, test_data )

if __name__ == '__main__':
  tf.app.run()

