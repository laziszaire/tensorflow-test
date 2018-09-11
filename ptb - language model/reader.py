import os
from collections import Counter
import tensorflow as tf

data_path = './ptb - language model/data/simple-examples/data'
train_path = os.path.join(data_path, 'ptb.train.txt')


def _build_vocab(doc):
    """

    :param doc: a list of words in doc
    :return:
    """
    counter = Counter(doc)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = zip(*count_pairs)
    word2id = dict(zip(words, range(len(words))))
    return word2id


def doc2ids(doc, word2id):
    return [word2id(w) for w in doc if w in word2id]


def ptb_raw_data(data_path=None):
    """
    transform word to ids in documents
    :param data_path:
    :return:
    """
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    with open(train_path, 'r') as f:
        train_doc = f.read().replace("\n", "<eos>").split()
        word2id = _build_vocab(train_doc)
        vocabulary = len(word2id)
    train_data = doc2ids(train_path, word2id)
    valid_data = doc2ids(valid_path, word2id)
    test_data = doc2ids(test_path, word2id)

    return train_data, valid_data, test_data, vocabulary

def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """

    :param raw_data: a list, one of train_data, valid_data or test_data
    :param batch_size:
    :param num_steps: number of unroll, truncated
    :param name: name of this operation
    :return: (inputs and target) 1 batch
            same shape: [batch_size, num_steps]
            the second element is time-shifted to right by one
    """
    with tf.name_scope(name, 'PTBProducer', [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name='raw_data', dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size*batch_len], [batch_size, batch_len])

        epoch_size = (batch_len-1) // num_steps # 每个batch_len里面有多少个num_steps

        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i*num_steps],
                             [batch_size, (i+1)*num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y







with open(train_path, 'r') as f:
    data = f.read().replace("\n", "<eos>").split()
    word2id = _build_vocab(data)
    id_doc = doc2ids(data, word2id)







