import tensorflow as tf
import numpy as np
from reader import ptb_producer, ptb_raw_data

class Flags:
    pass


class PTBinput:
    def __init__(self, data_path, batch_size, num_steps, name=None):
        raw_data = ptb_raw_data(data_path=data_path)
        train_data, valid_data, test_data, _ = raw_data
        self._dataset = ptb_producer(train_data, batch_size, num_steps, name=name)
        self._iterater = self._dataset.make_initializable_iterator()
        self.input_data, self.target = self._iterater.get_next()
        self.init = self._iterater.initializer
        self.batch_size = batch_size
        self.num_steps = num_steps


class LanguageModel:
    def __init__(self, is_training, config, input_):
        self._is_trainning = is_training
        self._input = input_
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self._final_state = None
        self._build()

    def add_embeddings(self):
        # hidden size 恰好等于 word embedding size
        embedding = tf.get_variable('embedding', [self.vocab_size, self.hidden_size])
        self.embedding = embedding
        inputs = tf.nn.embedding_lookup(embedding, self._input.input_data)
        return inputs

    def add_dropout(self, inputs):
        inputs = tf.nn.dropout(inputs, self.config.keep_prob)
        return inputs

    def get_cell(self):
        is_training = self._is_trainning
        h_size = self.hidden_size
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(h_size,
                                                 forget_bias=0.0,
                                                 state_is_tuple=True,
                                                 reuse=not is_training)
        if is_training and self.config.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=self.config.keep_prob)
        return lstm_cell

    def multi_cell(self):
        _multi_cell = tf.contrib.rnn.MultiRNNCell(
            [self.get_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)
        self._init_state = _multi_cell.zero_state(self.config.batch_size, self.data_type)
        return _multi_cell

    @property
    def data_type(self):
        return tf.float32

    def add_pred(self, inputs):
        with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE):
            self._cell = self.multi_cell()
            outputs, state = tf.nn.dynamic_rnn(self._cell, inputs, initial_state=self._init_state)
        output = tf.reshape(outputs, [-1, self.config.hidden_size])
        self._final_state = state
        return output, state

    def add_loss(self, output):
        softmax_w = tf.get_variable('softmax_w',
                                    shape=[self.hidden_size, self.config.vocab_size],
                                    dtype=self.data_type)
        softmax_b = tf.get_variable('softmax_b',
                                    shape=[self.config.vocab_size],
                                    dtype=self.data_type)
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, self.vocab_size])
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self._input.target,
            tf.ones([self.batch_size, self.num_steps], dtype=self.data_type),
            average_across_timesteps=False,
            average_across_batch=True
        )
        self._cost = tf.reduce_sum(loss)
        return self._cost

    def add_train_op(self, cost):
        self._lr = tf.Variable(0., trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.train.get_or_create_global_step())
        self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr)
        return self._train_op

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def _build(self):
        inputs = self.add_embeddings()
        inputs = self.add_dropout(inputs)
        output, _ = self.add_pred(inputs)
        cost = self.add_loss(output)
        self.add_train_op(cost)

    @property
    def cost(self):
        return self._cost

    @property
    def input(self):
        return self._input

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state(self):
        return self._init_state

    def run_epoch(self, session):
        costs, iters = 0, 0
        fetches = {'cost': self.cost,
                   'final_state': self._final_state,
                   'train': self.train_op,
                   'init_state': self.initial_state,
                   'lr': self._lr}
        session.run(tf.global_variables_initializer())
        self.assign_lr(session, 1)
        session.run(self.input.init)
        N_run = 0
        while True:
            try:
                vals = session.run(fetches)
                cost = vals['cost']
                costs += cost
                iters += self.input.num_steps
                if N_run % 10 == 1:
                    print(costs / iters)
                    print(vals['lr'])
                N_run += 1
            except tf.errors.OutOfRangeError:
                print('break')
                break
        print('done')
        return np.exp(costs / iters)

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def test_():
    """
    see if cost go down
    :return:
    """
    data_path = './data/simple-examples/data/'
    batch_size, num_steps = SmallConfig.batch_size, SmallConfig.num_steps
    input_ = PTBinput(data_path, batch_size=batch_size, num_steps=num_steps)
    config = SmallConfig
    is_training = True
    lm = LanguageModel(is_training, config, input_)
    with tf.Session() as sess:
        perplexity = lm.run_epoch(sess)
        print(perplexity)


if __name__ == "__main__":
    test_()















