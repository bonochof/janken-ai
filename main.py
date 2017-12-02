"""
  Deep Learning for Janken Game
  2017.12  Takata
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import datetime as dt
import time

class Janken:
  # initialize class
  def __init__( self, datanum=1000 ):
    self.datanum = datanum

  # create prob of opponent hand
  def opponent_hand( self ):
    rand_rock = np.random.rand()
    rand_scissors = np.random.rand()
    rand_paper = np.random.rand()
    total = rand_rock + rand_scissors + rand_paper
    return [rand_rock/total, rand_scissors/total, rand_paper/total]

  # return winnable hand
  def winning_hand( self, rock, scissors, paper ) -> [float, float, float]:
    mx = max( [rock, scissors, paper] )
    if mx == rock: return [0, 0, 1]
    if mx == scissors: return [1, 0, 0]
    if mx == paper: return [0, 1, 0]

  # create training data
  def get_training_data( self, n_data=None ):
    if n_data is None:
      n_data = self.datanum

    training_data_input = []
    training_data_output = []
    for i in range( n_data ):
      rock_prob, scissors_prob, paper_prob = self.opponent_hand()
      input_probs = [rock_prob, scissors_prob, paper_prob]
      training_data_input.append( input_probs )
      training_data_output.append( self.winning_hand( *input_probs ) )
    return {'input': training_data_input, 'output': training_data_output}

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer( 'test_data', 10, 'number of test data')
tf.app.flags.DEFINE_integer( 'training_data', 1000, 'number of training data')

def main( argv=None ):
  janken = Janken()
  training_data = janken.get_training_data( FLAGS.training_data )
  test_data = janken.get_training_data( FLAGS.test_data )

if __name__ == '__main__':
  tf.app.run()

