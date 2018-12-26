
import time
import os

import tensorflow as tf	
import numpy as np

from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT

from utils import TextLoader


class SampleLoader():
	def __init__(self, pos_loader, generator, generated_num, batch_size, sess):
		self.batch_size = batch_size

		positive_examples = []
		negative_examples = []

		# Positive Samples
		pos_loader.reset_batch_pointer()
		for _ in range(pos_loader.num_batches):
			x, _ = pos_loader.next_batch()
			positive_examples.extend(x)

		# Generate Samples
		for _ in range(int(generated_num / batch_size)):
			negative_examples.extend(generator.generate(sess))

		self.sentences = np.array(positive_examples + negative_examples)
		
		positive_labels = [[0, 1] for _ in positive_examples]
		negative_labels = [[1, 0] for _ in negative_examples]
		self.labels = np.concatenate([positive_labels, negative_labels], 0)

		# Shuffle the data
		shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
		self.sentences = self.sentences[shuffle_indices]
		self.labels = self.labels[shuffle_indices]

		# Split batches
		self.num_batch = int(len(self.labels) / self.batch_size)
		self.sentences = self.sentences[:self.num_batch * self.batch_size]
		self.labels = self.labels[:self.num_batch * self.batch_size]
		self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
		self.labels_batches = np.split(self.labels, self.num_batch, 0)
	
		self.pointer = 0

	def next_batch(self):
		ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batch
		return ret

	def reset_pointer(self):
		self.pointer = 0
		

data_dir = 'data/tinyshakespeare'
save_dir = 'models_seqgan'

###
BATCH_SIZE = 128
SEQ_LENGTH = 40
EMB_DIM = 96
HIDDEN_DIM = 64
START_TOKEN = 0
PRE_EPOCH_NUM = 120
#PRE_EPOCH_NUM = 1

###
dis_epoch_num = 50
#dis_epoch_num = 1
dis_embedding_dim = 128
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 128

###
TOTAL_BATCH = 200
#TOTAL_BATCH = 1
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 3000

def train():
	###
	if os.path.exists(save_dir):
		os.system('rm -rf {}'.format(save_dir))

	###
	data_loader = TextLoader(data_dir, BATCH_SIZE, SEQ_LENGTH)
	vocab_size = data_loader.vocab_size

	###
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	tf.reset_default_graph()

	###
	generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
	discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
			filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

	###
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.global_variables())

	####
	for epoch in range(PRE_EPOCH_NUM):
		supervised_g_losses = []
		data_loader.reset_batch_pointer()
		for _ in range(data_loader.num_batches):
			x, _ = data_loader.next_batch()
			_, g_loss = generator.pretrain_step(sess, x)
			supervised_g_losses.append(g_loss)
		loss = np.mean(supervised_g_losses)
		if epoch % 10 == 0:
			print('Generator pretrain : epoch {} - loss {:.4f}'.format(epoch, loss))

	####
	for epoch in range(dis_epoch_num):
		dis_data_loader = SampleLoader(data_loader, generator, generated_num, BATCH_SIZE, sess)
		for _ in range(3):
			dis_data_loader.reset_pointer()
			for it in range(dis_data_loader.num_batch):
				x_batch, y_batch = dis_data_loader.next_batch()
				feed = {
					discriminator.input_x: x_batch,
					discriminator.input_y: y_batch,
					discriminator.dropout_keep_prob: dis_dropout_keep_prob
				}
				_ = sess.run(discriminator.train_op, feed)
		if epoch % 10 == 0:
			print('Discriminator pretrain : epoch {}'.format(epoch))

	rollout = ROLLOUT(generator, 0.8)

	####
	for total_batch in range(TOTAL_BATCH):
		start_time = time.time()
		print('Adversal train : batch {} : START'.format(total_batch))
		# Train the generator for one step
		for it in range(1):
			samples = generator.generate(sess)
			rewards = rollout.get_reward(sess, samples, 16, discriminator)
			feed = {generator.x: samples, generator.rewards: rewards}
			_ = sess.run(generator.g_updates, feed_dict=feed)

		# Update roll-out parameters
		rollout.update_params()

		# Train the discriminator
		for _ in range(5):
			dis_data_loader = SampleLoader(data_loader, generator, generated_num, BATCH_SIZE, sess)
			for _ in range(3):
				dis_data_loader.reset_pointer()
				for it in range(dis_data_loader.num_batch):
					x_batch, y_batch = dis_data_loader.next_batch()
					feed = {
						discriminator.input_x: x_batch,
						discriminator.input_y: y_batch,
						discriminator.dropout_keep_prob: dis_dropout_keep_prob
					}
					_ = sess.run(discriminator.train_op, feed)

		if total_batch % 1 == 0:
			samples = generator.generate(sess)
			for i in range(20):
				print(''.join([data_loader.chars[sample] for sample in samples[i]]))

		if total_batch % 1 == 0:
			checkpoint_path = os.path.join(save_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=total_batch)
			print("model saved to {}".format(checkpoint_path))

		print('Adversal train : batch {} : END tooks {}'.format(total_batch, time.time() - start_time))

def test():
	###
	data_loader = TextLoader(data_dir, BATCH_SIZE, SEQ_LENGTH)
	vocab_size = data_loader.vocab_size

	###
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	tf.reset_default_graph()

	###
	generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
	discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
			filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

	###
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state(save_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		samples = generator.generate(sess)
		for i in range(20):
			print(''.join([data_loader.chars[sample] for sample in samples[i]]))

def main():
	train()
#test()

if __name__ == '__main__':
	main()
