import os
from collections import Counter
import tensorflow as tf
import tempfile
import numpy as np

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
    return [word2id[w] for w in doc if w in word2id]


def ptb_raw_data(data_path=None):
    """
    transform word to ids in documents
    :param data_path:
    :return:
    """
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    train_doc, valid_doc, test_doc = map(path2doc, [train_path, valid_path, test_path])
    word2id = _build_vocab(train_doc)
    vocabulary = len(word2id)
    train_data, valid_data, test_data = map(lambda x: doc2ids(x, word2id), [train_doc, valid_doc, test_doc])

    return train_data, valid_data, test_data, vocabulary


def path2doc(path):
    with open(path, 'r') as f:
        doc = f.read().replace("\n", "<eos>").split()
    return doc

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
    with tf.name_scope(name, 'PTB_Dataset', [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, dtype=tf.int32, name='raw_data')
        data_len = tf.size(raw_data)
        examples = tf.data.Dataset.range(0, tf.cast(data_len-1, tf.int64), num_steps)

        def _xy(i):
            return raw_data[i:i+num_steps], raw_data[i+1: i+num_steps+1]

        dataset = examples.map(_xy)
        return dataset.batch(batch_size, drop_remainder=True)


def test_ptb_raw_data():
    data_path = './data/simple-examples/data'
    output = ptb_raw_data(data_path)
    assert len(output) == 4
    print('raw data smoke test pass')


def test_ptb_producer():
    """
    simple example to test ptb reader
    :return:
    """
    raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
    batch_size = 1
    num_steps = 2
    dataset = ptb_producer(raw_data, batch_size, num_steps)
    iterator = dataset.make_one_shot_iterator()
    e = iterator.get_next()
    with tf.Session() as sess:
        x, y = sess.run(e)
        assert np.allclose(x, [[4, 3]])
        assert np.allclose(y, [[3, 2]])
        print('ptb dataset producer pass')


if __name__ == "__main__":
    test_ptb_producer()
    test_ptb_raw_data()







