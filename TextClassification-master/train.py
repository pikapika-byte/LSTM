# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import json
import datetime
import pickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import opt as tfopt

import data_helper
from rnn_classifier import rnn_clf
from cnn_classifier import cnn_clf
from clstm_classifier import clstm_clf

try:
    from sklearn.model_selection import train_test_split
except ImportError as e:
    error = "Please install scikit-learn."
    print(str(e) + ': ' + error)
    sys.exit()

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
# =============================================================================

# Model choices
tf.flags.DEFINE_string('clf', 'cnn', "Type of classifiers. Default: cnn. Choices: [cnn, lstm, blstm, clstm]")

# Optimization choices
tf.flags.DEFINE_string('optimizer', 'adam', "Optimizer algorithm. Choices: [gd, momentum, nag, rmsprop, adam, adamw, rnsa]")

# Data parameters
tf.flags.DEFINE_string('data_file', None, 'Data file path')
tf.flags.DEFINE_string('stop_word_file', None, 'Stop word file path')
tf.flags.DEFINE_string('language', 'en', "Language of the data file. Choices: [ch, en]")
tf.flags.DEFINE_integer('min_frequency', 0, 'Minimal word frequency')
tf.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.flags.DEFINE_integer('max_length', 0, 'Max document length')
tf.flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size')
tf.flags.DEFINE_float('test_size', 0.1, 'Cross validation test size')

# Model hyperparameters
tf.flags.DEFINE_integer('embedding_size', 256, 'Word embedding size.')
tf.flags.DEFINE_string('filter_sizes', '3, 4, 5', 'CNN filter sizes.')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size.')
tf.flags.DEFINE_integer('hidden_size', 128, 'Number of hidden units in LSTM.')
tf.flags.DEFINE_integer('num_layers', 2, 'Number of LSTM layers.')
tf.flags.DEFINE_float('keep_prob', 0.5, 'Dropout keep probability')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
tf.flags.DEFINE_float('l2_reg_lambda', 0.001, 'L2 regularization lambda')

# Training parameters
tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.flags.DEFINE_integer('num_epochs', 50, 'Number of epochs')
tf.flags.DEFINE_float('decay_rate', 1, 'Learning rate decay rate.')
tf.flags.DEFINE_integer('decay_steps', 100000, 'Learning rate decay steps')
tf.flags.DEFINE_integer('evaluate_every_steps', 100, 'Evaluate every steps')
tf.flags.DEFINE_integer('save_every_steps', 1000, 'Save model every steps')
tf.flags.DEFINE_integer('num_checkpoint', 10, 'Number of models to store')

FLAGS = tf.app.flags.FLAGS

# Parameter adjustments
if FLAGS.clf == 'lstm':
    FLAGS.embedding_size = FLAGS.hidden_size
elif FLAGS.clf == 'clstm':
    FLAGS.hidden_size = len(FLAGS.filter_sizes.split(",")) * FLAGS.num_filters

# Output files directory
timestamp = str(int(time.time()))
outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Load data
data, labels, lengths, vocab_processor = data_helper.load_data(file_path=FLAGS.data_file,
                                                               sw_path=FLAGS.stop_word_file,
                                                               min_frequency=FLAGS.min_frequency,
                                                               max_length=FLAGS.max_length,
                                                               language=FLAGS.language,
                                                               shuffle=True)

vocab_processor.save(os.path.join(outdir, 'vocab'))
FLAGS.vocab_size = len(vocab_processor.vocabulary_._mapping)
FLAGS.max_length = vocab_processor.max_document_length

params = FLAGS.flag_values_dict()
# Adjust params dict
model = params['clf']
if model == 'cnn':
    if 'hidden_size' in params: del params['hidden_size']
    if 'num_layers' in params: del params['num_layers']
elif model in ['lstm', 'blstm']:
    if 'num_filters' in params: del params['num_filters']
    if 'filter_sizes' in params: del params['filter_sizes']
    params['embedding_size'] = params['hidden_size']
elif model == 'clstm':
    params['hidden_size'] = len(list(map(int, params['filter_sizes'].split(",")))) * params['num_filters']

params_dict = sorted(params.items(), key=lambda x: x[0])
print('Parameters:')
for item in params_dict:
    print('{}: {}'.format(item[0], item[1]))
print('')

with open(os.path.join(outdir, 'params.pkl'), 'wb') as f:
    pkl.dump(params, f, True)

# Split data
x_train, x_valid, y_train, y_valid, train_lengths, valid_lengths = train_test_split(
    data, labels, lengths, test_size=FLAGS.test_size, random_state=22)

# Batch iterator
train_data = data_helper.batch_iter(x_train, y_train, train_lengths, FLAGS.batch_size, FLAGS.num_epochs)

# =============================================================================
# TRAIN
# =============================================================================

with tf.Graph().as_default():
    with tf.Session() as sess:
        if FLAGS.clf == 'cnn':
            classifier = cnn_clf(FLAGS)
        elif FLAGS.clf in ['lstm', 'blstm']:
            classifier = rnn_clf(FLAGS)
        elif FLAGS.clf == 'clstm':
            classifier = clstm_clf(FLAGS)
        else:
            raise ValueError('clf should be one of [cnn, lstm, blstm, clstm]')

        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        starter_learning_rate = FLAGS.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)

        # ---------------------------------------------------------------------
        # Optimization Logic
        # ---------------------------------------------------------------------
        opt_name = FLAGS.optimizer.lower()
        print("Using Optimizer: {}".format(opt_name.upper()))

        # Optimizer selection
        if opt_name == 'gd' or opt_name == 'rnsa':
            # RNSA uses GD as the base stepper
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif opt_name == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        elif opt_name == 'nag':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
        elif opt_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, epsilon=1e-8)
        elif opt_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif opt_name == 'adamw':
            optimizer = tfopt.AdamWOptimizer(weight_decay=FLAGS.l2_reg_lambda, 
                                             learning_rate=learning_rate, 
                                             beta1=0.9, beta2=0.999, epsilon=1e-8)
        else:
            raise ValueError("Unknown optimizer")

        # Compute Gradients
        grads_and_vars = optimizer.compute_gradients(classifier.cost)
        # Separate grads and vars for RNSA manual handling
        grads_only = [g for g, v in grads_and_vars]
        vars_only = [v for g, v in grads_and_vars]
        
        # Train Op
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # ---------------------------------------------------------------------
        # Helper Ops for RNSA (Variable Swapping)
        # ---------------------------------------------------------------------
        # Placeholders to load numpy arrays into TF variables
        var_placeholders = [tf.placeholder(v.dtype, shape=v.get_shape()) for v in vars_only]
        # Assign ops
        assign_ops = [tf.assign(v, p) for v, p in zip(vars_only, var_placeholders)]

        def set_model_params(params_list):
            """Assign numpy arrays to model variables."""
            feed = {p: v for p, v in zip(var_placeholders, params_list)}
            sess.run(assign_ops, feed_dict=feed)

        def get_model_params():
            """Get current model variables as numpy arrays."""
            return sess.run(vars_only)

        # ---------------------------------------------------------------------
        # Summaries
        # ---------------------------------------------------------------------
        loss_summary = tf.summary.scalar('Loss', classifier.cost)
        accuracy_summary = tf.summary.scalar('Accuracy', classifier.accuracy)

        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(outdir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        valid_summary_op = tf.summary.merge_all()
        valid_summary_dir = os.path.join(outdir, 'summaries', 'valid')
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoint)

        # Fix deprecated warning for init
        init_op = tf.compat.v1.global_variables_initializer() if hasattr(tf.compat.v1, 'global_variables_initializer') else tf.global_variables_initializer()
        sess.run(init_op)

        # ---------------------------------------------------------------------
        # RNSA State Variables
        # ---------------------------------------------------------------------
        rnsa_state = {
            'r_prev': 0.0,
            'theta_hat': None,  # This will store the "Output" parameters
            'active': (opt_name == 'rnsa')
        }

        def run_step(input_data, is_training=True):
            """Run one step of the training process."""
            input_x, input_y, sequence_length = input_data

            # Base Feed Dict
            feed_dict = {
                classifier.input_x: input_x,
                classifier.input_y: input_y,
                classifier.keep_prob: FLAGS.keep_prob if is_training else 1.0
            }
            if FLAGS.clf != 'cnn':
                feed_dict[classifier.batch_size] = len(input_x)
                feed_dict[classifier.sequence_length] = sequence_length

            # -----------------------------------------------------------------
            # RNSA Training Logic (Strict Implementation)
            # -----------------------------------------------------------------
            if is_training and rnsa_state['active']:
                # 1. Capture theta (theta_prev_gd in Matlab code)
                theta_prev_gd = get_model_params()

                # 2. Run GD step (updates theta -> theta_new_gd)
                # We also fetch cost/acc for logging the "Base" performance
                _, step, cost, accuracy = sess.run(
                    [train_op, global_step, classifier.cost, classifier.accuracy],
                    feed_dict=feed_dict
                )
                
                # 3. Capture theta_new_gd (theta_new in Matlab code)
                theta_new_gd = get_model_params()

                # 4. Calculate Gradient at NEW position (grad_new)
                # Need to run gradients on the SAME batch with NEW weights
                grad_vals_new = sess.run(grads_only, feed_dict=feed_dict)

                # 5. Calculate Norm of grad_new (r_curr)
                # FIX: Handle IndexedSlicesValue for Embedding layers
                flat_grads = []
                for g in grad_vals_new:
                    if hasattr(g, 'values'): 
                        # This is a sparse gradient (IndexedSlicesValue)
                        flat_grads.append(g.values.flatten())
                    else:
                        # This is a dense gradient (numpy array)
                        flat_grads.append(g.flatten())
                
                all_grads_flat = np.concatenate(flat_grads)
                r_curr = np.linalg.norm(all_grads_flat)

                # 6. RNSA Update Logic
                if step <= 1: 
                    rnsa_state['r_prev'] = r_curr
                    rnsa_state['theta_hat'] = theta_new_gd # Fallback to GD for first step
                    eta_k = 0.0
                    w_k = 0.0
                else:
                    # w_k = r_curr / r_prev
                    denominator = rnsa_state['r_prev'] + 1e-16
                    w_k = r_curr / denominator
                    
                    # eta_k = 1 / (1 - w_k)
                    eta_k = 1.0 / (1.0 - w_k )

                    # theta_hat = theta_prev_gd + eta_k * (theta_new_gd - theta_prev_gd)
                    theta_hat = []
                    for t_prev, t_new in zip(theta_prev_gd, theta_new_gd):
                        val = t_prev + eta_k * (t_new - t_prev)
                        theta_hat.append(val)
                    
                    rnsa_state['theta_hat'] = theta_hat
                    rnsa_state['r_prev'] = r_curr

                # Logging
                time_str = datetime.datetime.now().isoformat()
                print("{}: step: {}, loss(gd): {:g}, acc(gd): {:g}, w_k: {:.4f}, eta: {:.4f}".format(
                    time_str, step, cost, accuracy, w_k, eta_k))
                
                # Write summaries (using GD values for speed)
                current_summaries = sess.run(train_summary_op, feed_dict=feed_dict)
                train_summary_writer.add_summary(current_summaries, step)

                return accuracy

            # -----------------------------------------------------------------
            # Standard Optimizer Logic (Non-RNSA)
            # -----------------------------------------------------------------
            else:
                fetches = {'step': global_step,
                           'cost': classifier.cost,
                           'accuracy': classifier.accuracy}
                
                if is_training:
                    fetches['train_op'] = train_op
                    fetches['summaries'] = train_summary_op
                else:
                    fetches['summaries'] = valid_summary_op

                vars_res = sess.run(fetches, feed_dict)
                step = vars_res['step']
                cost = vars_res['cost']
                accuracy = vars_res['accuracy']
                summaries = vars_res['summaries']

                if is_training:
                    train_summary_writer.add_summary(summaries, step)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step: {}, loss: {:g}, acc: {:g}".format(time_str, step, cost, accuracy))
                else:
                    valid_summary_writer.add_summary(summaries, step)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step: {}, loss: {:g}, acc: {:g} (Validation)".format(time_str, step, cost, accuracy))

                return accuracy

        # Main Loop
        print('Start training with {} optimizer...'.format(FLAGS.optimizer))

        for train_input in train_data:
            run_step(train_input, is_training=True)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every_steps == 0:
                print('\nValidation')
                
                # If RNSA is active, we must evaluate theta_hat (the output), not theta (the state)
                gd_params = None
                if rnsa_state['active'] and rnsa_state['theta_hat'] is not None:
                    # 1. Save current GD params
                    gd_params = get_model_params()
                    # 2. Load theta_hat
                    set_model_params(rnsa_state['theta_hat'])
                    print("(Evaluating RNSA theta_hat)")
                
                run_step((x_valid, y_valid, valid_lengths), is_training=False)
                
                # Restore GD params to continue training
                if gd_params is not None:
                    set_model_params(gd_params)
                
                print('')

            if current_step % FLAGS.save_every_steps == 0:
                # Same logic: Save theta_hat if RNSA
                gd_params = None
                if rnsa_state['active'] and rnsa_state['theta_hat'] is not None:
                    gd_params = get_model_params()
                    set_model_params(rnsa_state['theta_hat'])
                    print("(Saving RNSA theta_hat)")

                save_path = saver.save(sess, os.path.join(outdir, 'model/clf'), current_step)
                
                if gd_params is not None:
                    set_model_params(gd_params)

        print('\nAll the files have been saved to {}\n'.format(outdir))