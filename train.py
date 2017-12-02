"""
  Deep Learning for Janken Game using MLP
  train.py
  2017.12  Takata
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import datetime as dt
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string( 'summary-dir', '/tmp/tensorflow/summary', 'log files path' )
tf.app.flags.DEFINE_integer( 'max-epoch', 100, 'max epoch number' )
tf.app.flags.DEFINE_integer( 'batch-size', 10, 'batch size' )
tf.app.flags.DEFINE_float( 'learning-rate', 0.07, 'learning rate' )
tf.app.flags.DEFINE_integer( 'test-data', 10, 'test data number' )
tf.app.flags.DEFINE_integer( 'training-data', 1000, 'training data number' )
tf.app.flags.DEFINE_boolean( 'skip-training', False, 'check to test without learning' )

def train_and_test( training_data, test_data ):
  if len( training_data['input'] ) != len( training_data['output'] ):
    print( "Error: input number does not agree with output number (training data)" )
    return
  if len( test_data['input'] ) != len( test_data['output'] ):
    print( "Error: input number does not agree with output number (test data)" )
    return

  # create input layer
  with tf.name_scope( 'Inputs' ):
    input = tf.placeholder( tf.float32, shape=[None, 3], name='Input' )
  with tf.name_scope( 'Outputs' ):
    true_output = tf.placeholder( tf.float32, shape=[None, 3], name='Output' )

  # create hidden layer
  def hidden_layer( x, layer_size, is_output=False ):
    name = 'Hidden-Layer' if not is_output else 'Output-Layer'
    with tf.name_scope( name ):
      # weight
      w = tf.Variable( tf.random_normal( [x.shape[1].value, layer_size] ), name='Weight' )
      # bias
      b = tf.Variable( tf.zeros( [layer_size] ), name='Bias' )
      # sum
      z = tf.matmul( x, w ) + b
      a = tf.tanh( z ) if not is_output else z
    return a

  # make 3-10-10-3 layer
  layer1 = hidden_layer( input, 10 )
  layer2 = hidden_layer( layer1, 10 )
  output = hidden_layer( layer2, 3, is_output=True )

  # define loss
  with tf.name_scope( "Loss" ):
    with tf.name_scope( "Cross-Entropy" ):
      error = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels=true_output, logits=output ) )
    with tf.name_scope( "Accuracy" ):
      accuracy = tf.reduce_mean( tf.cast( tf.equal( tf.arg_max( true_output, 1 ), tf.argmax( output, 1 ) ), tf.float32 ) ) * 100.0
    with tf.name_scope( "Prediction" ):
      prediction = tf.nn.softmax( output )

  # create op of training
  with tf.name_scope( "Train" ):
    train_op = tf.train.GradientDescentOptimizer( FLAGS.learning_rate ).minimize( error )

  # create session
  sess = tf.Session()
  sess.run( tf.global_variables_initializer() )

  # summary of tensorboard
  writer = tf.summary.FileWriter( FLAGS.summary_dir + '/' + dt.datetime.now().strftime('%Y%m%d-%H%M%S'), sess.graph )
  tf.summary.scalar( 'CrossEntropy', error )
  tf.summary.scalar( 'Accuracy', accuracy )
  summary = tf.summary.merge_all()

  # learning
  def train():
    print( '------------------------------ Start Learning -----------------------------' )
    batch_size = FLAGS.batch_size
    loop_per_epoch = int( len( training_data['input'] ) / batch_size )
    max_epoch = FLAGS.max_epoch
    print_interval = max_epoch / 10 if max_epoch >= 10 else 1
    step = 0
    start_time = time.time()
    for e in range( max_epoch ):
      for i in range( loop_per_epoch ):
        batch_input = training_data['input'][i*batch_size:(i+1)*batch_size]
        batch_output = training_data['output'][i*batch_size:(i+1)*batch_size]
        _, loss, acc, report = sess.run( [train_op, error, accuracy, summary], feed_dict={input: batch_input, true_output: batch_output} )
        step += batch_size

      writer.add_summary( report, step )
      writer.flush()

      if ( e + 1 ) % print_interval == 0:
        learning_speed = ( e + 1.0 ) / ( time.time() - start_time )
        print( 'epoch:{:3}  cross_entropy:{:.6f}  accuracy:{:6.2f}%  speed:{:5.2f}[epoch/sec]'.format( e+1, loss, acc, learning_speed ) )

    print( '------------------------------ End Learning ------------------------------' )
    print( 'time spent on {} epoch: {:.2f}[sec]'.format( max_epoch, time.time() - start_time ) )

  # test learning result
  def test():
    print( '------------------------------ Start Trial ------------------------------' )
    print( '{:5}        {:20}          {:20}            {:20}{:2}'.format( '', 'opponent hand', 'winnable hand', 'AI hand', 'result' ) )
    print( '{}   {:3}   {:3}  {:3}        {:3}   {:3}  {:3}        {:3}   {:3}  {:3}'.format( 'No.  ', 'Rock', 'Scissors', 'Paper', 'Rock', 'Scissors', 'Paper', 'Rock', 'Scissors', 'Paper' ) )

    # display highlight
    def highlight( rock, scissors, paper ):
      mx = max( rock, scissors, paper )
      rock_prob_em = '[{:6.4f}]'.format( rock ) if mx == rock else '{:^8.4f}'.format( rock )
      scissors_prob_em = '[{:6.4f}]'.format( scissors ) if mx == scissors else '{:^8.4f}'.format( scissors )
      paper_prob_em = '[{:6.4f}]'.format( paper ) if mx == paper else '{:^8.4f}'.format( paper )
      return [rock_prob_em, scissors_prob_em, paper_prob_em]

    # test
    win_count = 0
    for k in range( len( test_data['input'] ) ):
      input_probs = [test_data['input'][k]]
      output_probs = [test_data['output'][k]]

      # run test operation
      acc, predict = sess.run( [accuracy, prediction], feed_dict={input: input_probs, true_output: output_probs} )

      best_bet_label = np.argmax( output_probs, 1 )
      best_bet_logit = np.argmax( predict, 1 )
      result = 'lose'
      if best_bet_label == best_bet_logit:
        win_count += 1
        result = 'win'

      print( '{:<5} {:8} {:8} {:8}'.format( *( tuple( [k+1]+highlight( *input_probs[0] ) ) ) ), end='' )
      print( '    ', end='' )
      print( '{:8} {:8} {:8}'.format( *tuple( highlight( *output_probs[0] ) ) ) , end='' )
      print( '    ', end='' )
      print( '{:8} {:8} {:8}'.format( *tuple( highlight( *predict[0] ) ) ), end='' )
      print( '    ', end='' )
      print( '{:2}'.format( result ) )

    print( '------------------------------ End Trial ------------------------------' )
    print( 'win: {} lose: {} win rate: {:4.3f}%'.format( win_count, FLAGS.test_data-win_count, ( win_count / len( test_data['input'] ) * 100.0 ) ) )

  print( 'Test without learning' )
  test()

  if not FLAGS.skip_training:
    train()
    print( 'Test with learning' )
    test()

