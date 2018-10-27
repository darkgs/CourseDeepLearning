
from __future__ import print_function
import numpy as np
import random

from rnn_layers import *

import tensorflow as tf
import time, os, json, sys
import numpy as np
import matplotlib.pyplot as plt

#import pandas as pd

import tensorflow.python.platform
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import argparse

from six.moves import cPickle
from six import text_type

# this module is from the utils.py file of this folder
# it handles loading texts to digits (aka. tokens) which are recognizable for the model
from utils import TextLoader

from coco_utils import *
from captioning import *

# this module is from the char_rnn.py file of this folder
# the task is implementing the CharRNN inside the class definition from this file
from char_rnn import Model

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-g', '--gpu_num', dest='gpu_num', type='string', default='0')


def part_1():
    # implement rnn_step_forward
    # errors should be less than 1e-8
    x, prev_h, Wx, Wh, b, expected_next_h = getdata_rnn_step_forward()

    next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)

    print('next_h error: ', rel_error(expected_next_h, next_h))

    ##############################################################################

    # implement rnn_step_backward
    # errors should be less than 1e-8
    np.random.seed(2177)

    x, h, Wx, Wh, b, expected_next_h = getdata_rnn_step_forward()
    out, cache = rnn_step_forward(x, h, Wx, Wh, b)
    dnext_h = np.random.randn(*out.shape)
    dx_num, dprev_h_num, dWx_num, dWh_num, db_num = getdata_rnn_step_backward(x,h,Wx,Wh,b,dnext_h)

    dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)

    print('dx error     : ', rel_error(dx_num, dx))
    print('dprev_h error: ', rel_error(dprev_h_num, dprev_h))
    print('dWx error    : ', rel_error(dWx_num, dWx))
    print('dWh error    : ', rel_error(dWh_num, dWh))
    print('db error     : ', rel_error(db_num, db))

    ##############################################################################

    # implement rnn_forward
    # errors should be less than 1e-7
    x,h0,Wx,Wh,b, expected_h = getdata_rnn_forward()

    h, _ = rnn_forward(x, h0, Wx, Wh, b)

    print('h error: ', rel_error(expected_h, h))

    ##############################################################################

    # implement rnn_forward
    # errors should be less than 1e-7
    np.random.seed(2177)

    x,h0,Wx,Wh,b, expected_h = getdata_rnn_forward()
    out, cache = rnn_forward(x, h0, Wx, Wh, b)
    dout = np.random.randn(*out.shape)
    dx_num, dh0_num, dWx_num, dWh_num, db_num = getdata_rnn_backward(x,h0,Wx,Wh,b,dout)

    dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)

    print('dx error: ', rel_error(dx_num, dx))
    print('dh0 error: ', rel_error(dh0_num, dh0))
    print('dWx error: ', rel_error(dWx_num, dWx))
    print('dWh error: ', rel_error(dWh_num, dWh))
    print('db error: ', rel_error(db_num, db))


def part_2(gpu_num):

    model_path='./models_captioning'
    data_path ='./coco/coco_captioning'

    data = load_coco_data(base_dir=data_path)
    if len(data)==8 : 
        print('COCO data load complete')
        
    for k, v in data.items():
        if type(v) == np.ndarray:
            print(k, type(v), v.shape, v.dtype)
        else:
            print(k, type(v), len(v))

#train_data = load_coco_data(base_dir=data_path, max_train=512*100)
    train_data = data

    captions = train_data['train_captions']
    img_idx  = train_data['train_image_idxs']
    img_features = train_data['features'][img_idx]
    word_to_idx = train_data['word_to_idx']
    idx_to_word = train_data['idx_to_word']

    n_words = len(word_to_idx)
    maxlen = train_data['train_captions'].shape[1]
    input_dimension = train_data['features'].shape[1]

    print(n_words)
    print(maxlen)
    print(img_features.shape, captions.shape)

    vcaptions = train_data['val_captions']
    vimg_idx  = train_data['val_image_idxs']
    vimg_features = train_data['features'][vimg_idx]
    print(vimg_features.shape, vcaptions.shape)

    def train(img_features, captions):
        #---------------------------------------------------------
        # You must show maxlen, n_words, caption.shap
        #---------------------------------------------------------
        print(maxlen, n_words, captions.shape)

        #################################################
        # TODO: Implement caption training
        # - save your trained model in model_path
        # - must print about 10 loss changes !!
        #################################################
        img_feature_dim = img_features.shape[1]
        batch_size = 512
        n_epochs = 100

        conf = tf.ConfigProto()
        conf.gpu_options.per_process_gpu_memory_fraction = 0.2
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

        def calc_caption_length(caption):
            for i, word_idx in enumerate(caption):
                if word_idx == 0:
                    return min(len(caption), i+2)
            return len(caption)
        caption_lens = np.array([calc_caption_length(caption) for caption in captions])
        vcaption_lens = np.array([calc_caption_length(caption) for caption in vcaptions])

        known_caption_idxs = np.all(captions != 3, axis=1)

        known_captions = captions[known_caption_idxs]
        known_caption_lens = caption_lens[known_caption_idxs]
        known_img_features = img_features[known_caption_idxs]

        tf.reset_default_graph()

        if gpu_num == '0':
            captioning = Captioning(img_feature_dim, maxlen, n_words, 386, 3)
        elif gpu_num == '1':
            captioning = Captioning(img_feature_dim, maxlen, n_words, 848, 3)
        elif gpu_num == '2':
            captioning = Captioning(img_feature_dim, maxlen, n_words, 496, 3)
        elif gpu_num == '3':
            captioning = Captioning(img_feature_dim, maxlen, n_words, 612, 3)
        input_sequences, input_sequence_lens, input_img_features, \
            loss, optimize, drop_out_keep_rate, teacher_force = captioning.build_model()
        predict_indices = captioning.predict()

        # Saver
        saved_model_path = 'saved_model_{}/ass3_show_and_tell.ckpt'.format(gpu_num)
        epoch_save_path = 'saved_model/ass3_epoch.txt'
        min_valid_loss = 9999.99
        max_BLEU = -1.0
        saver = tf.train.Saver()
        with tf.Session(config=conf) as sess:
            sess.run(tf.global_variables_initializer())

            def generate_minibatch():
                permutation = np.random.permutation(len(known_captions))
                return known_captions[permutation][:batch_size], \
                    known_caption_lens[permutation][:batch_size], \
                    known_img_features[permutation][:batch_size]

            def generate_minibatch_valid(data_size):
                permutation = np.random.permutation(data_size)
                return vcaptions[permutation][:batch_size], \
                    vcaption_lens[permutation][:batch_size], \
                    vimg_features[permutation][:batch_size]

            for epoch in range(n_epochs):
                start_time = time.time()
                data_size = captions.shape[0]
                steps = data_size // batch_size

                # train
                loss_sum = 0.0
                for step in range(steps):
                    input_seqs, input_seq_lens, input_imgs = generate_minibatch()
                    feed_dict = {
                        input_sequences: input_seqs,
                        input_sequence_lens: input_seq_lens,
                        input_img_features: input_imgs,
                        drop_out_keep_rate: 0.1,
                    }
                    train_loss, _, train_force = sess.run([loss, optimize, teacher_force], feed_dict=feed_dict)
                    loss_sum += train_loss

                # validation
                val_loss_sum = 0.0
                val_data_size = vcaptions.shape[0]
#                val_steps = val_data_size // batch_size
#                for step in range(val_steps):
#                    input_seqs, input_seq_lens, input_imgs = generate_minibatch_valid(val_data_size)
#                    feed_dict = {
#                        input_sequences: input_seqs,
#                        input_sequence_lens: input_seq_lens,
#                        input_img_features: input_imgs,
#                    }      
#                    val_loss, val_force = sess.run([loss, teacher_force], feed_dict=feed_dict)
#                    val_loss_sum += val_loss


                # test sentence from validation
                pick_idx = random.randrange(val_data_size)
                feed_dict = {
                    input_img_features: np.expand_dims(img_features[pick_idx], axis=0),
                }

                pred_indices = sess.run([predict_indices], feed_dict=feed_dict)
                print([idx_to_word[idx] for idx in pred_indices[0][0]])

                # BLEU score
                def image_captioning(features) :
                    pr_captions = np.zeros((features.shape[0],maxlen),int)
                    feed_dict = {
                        input_img_features: features,
                    }
                    predicts = sess.run([predict_indices], feed_dict=feed_dict)
                    pr_captions = np.array(predicts[0])
                    return pr_captions

                def evaluate_model(data, split):
                    BLEUscores = {}

                    minibatch = sample_coco_minibatch(data, split=split, batch_size="All")
                    gt_captions, features, urls = minibatch
                    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

                    pr_captions = image_captioning(features)
                    pr_captions = decode_captions(pr_captions, data['idx_to_word'])

                    total_score = 0.0
                    
                    for gt_caption, pr_caption, url in zip(gt_captions, pr_captions, urls):
                        total_score += BLEU_score(gt_caption, pr_caption)

                    BLEUscores[split] = total_score / len(pr_captions)

                    return BLEUscores[split]

                cur_BLEU = evaluate_model(train_data, 'val')

                print('epoch {} : '.format(epoch) + ('train loss %.4f, ' % (loss_sum / steps)) + \
                        ('BLEU score %.4f' % cur_BLEU))
        
                if cur_BLEU > max_BLEU:
                    max_BLEU = cur_BLEU
                    saver.save(sess, saved_model_path)
                    print('model saved!!')

                if (epoch > 4) and (max_BLEU * 0.9) > cur_BLEU:
                    print('early stop!!')
                    break

        #print("Current Cost: ", loss_value, "\t Epoch {}/{}".format(epoch, n_epochs))
        
        #################################################
        #                END OF YOUR CODE               #
        #################################################

    train(img_features, captions) 

def part_3(gpu_num):
    # for TensorFlow vram efficiency: if this is not specified, the model hogs all the VRAM even if it's not necessary
    # bad & greedy TF! but it has a reason for this design choice FWIW, try googling it if interested
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

    # argparsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data and model checkpoints directories
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                                help='data directory containing input.txt with training examples')
    parser.add_argument('--save_dir', type=str, default='models_char_rnn',
                                help='directory to store checkpointed models')
    parser.add_argument('--save_every', type=int, default=1000,
                                help='Save frequency. Number of passes between checkpoints of the model.')
    parser.add_argument('--init_from', type=str, default=None,
                                help="""continue training from saved model at this path (usually "save").
                                     Path must contain files saved by previous training process:
                                     'config.pkl'        : configuration;
                                     'chars_vocab.pkl'   : vocabulary definitions;
                                     'checkpoint'        : paths to model file(s) (created by tf).
                                     Note: this file contains absolute paths, be careful when moving files around;
                                     'model.ckpt-*'      : file(s) with model definition (created by tf)
                                     Model params must be the same between multiple runs (model, rnn_size, num_layers and seq_length).
                                     """)
    # Model params
    parser.add_argument('--model', type=str, default='lstm',
                                help='lstm, rnn, gru, or nas')
    parser.add_argument('--rnn_size', type=int, default=128,
                                help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=3,
                                help='number of layers in the RNN')
    # Optimization
    parser.add_argument('--seq_length', type=int, default=500,
                                help='RNN sequence length. Number of timesteps to unroll for.')
    parser.add_argument('--batch_size', type=int, default=128,
                                help="""minibatch size. Number of sequences propagated through the network in parallel.
                                        Pick batch-sizes to fully leverage the GPU (e.g. until the memory is filled up)
                                        commonly in the range 10-500.""")
    parser.add_argument('--num_epochs', type=int, default=50,
                                help='number of epochs. Number of full passes through the training examples.')
    parser.add_argument('--grad_clip', type=float, default=20.,
                                help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                                help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=0.1,
                                help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=0.1,
                                help='probability of keeping weights in the input layer')

    # needed for argparsing within jupyter notebook
    # https://stackoverflow.com/questions/30656777/how-to-call-module-written-with-argparse-in-ipython-notebook
    sys.argv = ['-f']
    args = parser.parse_args()

    if gpu_num == '0':
        args.save_dir = 'models_char_rnn_{}'.format(gpu_num)
        args.init_from = args.save_dir
        args.rnn_size = 96
        args.num_layers = 3
        args.decay_rate = 0.99
        args.output_keep_prob = 0.1
        args.input_keep_prob = 0.1
        args.grad_clip = 5.0
        args.num_epochs = 10
    elif gpu_num == '1':
        args.save_dir = 'models_char_rnn_{}'.format(gpu_num)
        args.init_from = args.save_dir
        args.rnn_size = 128
        args.num_layers = 3
        args.decay_rate = 0.99
        args.output_keep_prob = 0.1
        args.input_keep_prob = 0.1
        args.num_epochs = 10
    elif gpu_num == '2':
        args.save_dir = 'models_char_rnn_{}'.format(gpu_num)
        args.init_from = args.save_dir
        args.rnn_size = 312
        args.num_layers = 3
        args.decay_rate = 0.99
        args.output_keep_prob = 0.1
        args.input_keep_prob = 0.1
        args.num_epochs = 10
    elif gpu_num == '3':
        args.save_dir = 'models_char_rnn_{}'.format(gpu_num)
        args.init_from = args.save_dir
        args.rnn_size = 396
        args.num_layers = 3
        args.decay_rate = 0.99
        args.output_keep_prob = 0.1
        args.input_keep_prob = 0.1
        args.num_epochs = 10

    # protip: always check the data and poke around the data yourself
    # you will get a lot of insights by looking at the data
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    data_loader.reset_batch_pointer()

    # training loop definition
    def train(args):
        data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
        args.vocab_size = data_loader.vocab_size
        print("vocabulary size: " + str(args.vocab_size))

        # check compatibility if training is continued from previously saved model
        if args.init_from is not None:
            # check if all necessary files exist
            assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
            assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
            assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
            ckpt = tf.train.latest_checkpoint(args.init_from)
            assert ckpt, "No checkpoint found"

            # open old config and check if models are compatible
            with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
                saved_model_args = cPickle.load(f)
            need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
            for checkme in need_be_same:
                assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

            # open saved vocab/dict and check if vocabs/dicts are compatible
            with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
                saved_chars, saved_vocab = cPickle.load(f)
            assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
            assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
            cPickle.dump(args, f)
        with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
            cPickle.dump((data_loader.chars, data_loader.vocab), f)
                                                                        
        print("building the model... may take some time...")
        ##################### This line builds the CharRNN model defined in char_rnn.py #####################
        tf.reset_default_graph()
        model = Model(args)
        print("model built! starting training...")

        with tf.Session(config=conf) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            # restore model
            if args.init_from is not None:
                saver.restore(sess, ckpt)
            for e in range(args.num_epochs):
                sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
                data_loader.reset_batch_pointer()
                state = sess.run(model.initial_state)
                                        
                for b in range(int(data_loader.num_batches)):
                    start = time.time()
                    x, y = data_loader.next_batch()
                    feed = {model.input_data: x, model.targets: y}
                    for i, (c, h) in enumerate(model.initial_state):
                        feed[c] = state[i].c
                        feed[h] = state[i].h

                    train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)

                    end = time.time()
                    
                    # print training log every 100 steps
                    if ((e * data_loader.num_batches + b) % 100 == 0):
                        print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                                    .format(e * data_loader.num_batches + b,
                                    args.num_epochs * data_loader.num_batches,
                                    e, train_loss, end - start))
                    if (e * data_loader.num_batches + b) % args.save_every == 0\
                                    or (e == args.num_epochs-1 and b == data_loader.num_batches-1):
                        # save for the last result
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                        print("model saved to {}".format(checkpoint_path))

    for i in range(10):
        train(args)
        print('==============================={}==================================='.format(i))
        part_3_sample(gpu_num)
        print('====================================================================')


def part_3_sample(gpu_num):
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

    def sample_eval(args):
        with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
            chars, vocab = cPickle.load(f)
        #Use most frequent char if no prime is given
        if args.prime == '':
            args.prime = chars[0]
        tf.reset_default_graph()
        model = Model(saved_args, training=False)

        with tf.Session(config=conf) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print(str(model.sample(sess, chars, vocab, args.n, args.prime)),'utf-8')

    parser_sample = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_sample.add_argument('--save_dir', type=str, default='models_char_rnn',
            help='model directory to store checkpointed models')
    parser_sample.add_argument('-n', type=int, default=500,
            help='number of characters to sample')
    parser_sample.add_argument('--prime', type=text_type, default=u'',
            help='prime text')
    sys.argv = ['-f']

    args_sample = parser_sample.parse_args()
    args_sample.save_dir = 'models_char_rnn_{}'.format(gpu_num)

    sample_eval(args_sample)


def main():
    options, args = parser.parse_args()
    gpu_num = options.gpu_num

    part_3(gpu_num)

if __name__ == '__main__':
    main()

