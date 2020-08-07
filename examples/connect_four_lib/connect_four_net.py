import numpy as np
import tensorflow as tf
import os

"""
TODO:
Refactor PyO3 bindings so that this is all a class
and not just a bunch of functions that get called
"""

saver = None
session = None
epochs = 10
batch_size = 64
dropout = 0.3

import tensorflow as tf


class Connect4NNet:
    def __init__(self, lr=0.001, num_channels=512):
        self.board_x, self.board_y = (7, 6)
        self.action_size = 6
        self.lr = lr
        self.num_channels = num_channels

        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense

        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_boards = tf.placeholder(
                tf.float32, shape=[None, self.board_x, self.board_y]
            )  # s: batch_size x board_x x board_y
            self.dropout = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool, name="is_training")

            x_image = tf.reshape(
                self.input_boards, [-1, self.board_x, self.board_y, 2]
            )  # batch_size  x board_x x board_y x 2
            h_conv1 = Relu(
                BatchNormalization(
                    self.conv2d(x_image, self.num_channels, "same"),
                    axis=3,
                    training=self.is_training,
                )
            )  # batch_size  x board_x x board_y x num_channels
            h_conv2 = Relu(
                BatchNormalization(
                    self.conv2d(h_conv1, self.num_channels, "same"),
                    axis=3,
                    training=self.is_training,
                )
            )  # batch_size  x board_x x board_y x num_channels
            h_conv3 = Relu(
                BatchNormalization(
                    self.conv2d(h_conv2, self.num_channels, "valid"),
                    axis=3,
                    training=self.is_training,
                )
            )  # batch_size  x (board_x-2) x (board_y-2) x num_channels
            h_conv4 = Relu(
                BatchNormalization(
                    self.conv2d(h_conv3, self.num_channels, "valid"),
                    axis=3,
                    training=self.is_training,
                )
            )  # batch_size  x (board_x-4) x (board_y-4) x num_channels
            h_conv4_flat = tf.reshape(
                h_conv4,
                [-1, self.num_channels * (self.board_x - 4) * (self.board_y - 4)],
            )
            s_fc1 = Dropout(
                Relu(
                    BatchNormalization(
                        Dense(h_conv4_flat, 1024), axis=1, training=self.is_training
                    )
                ),
                rate=self.dropout,
            )  # batch_size x 1024
            s_fc2 = Dropout(
                Relu(
                    BatchNormalization(
                        Dense(s_fc1, 512), axis=1, training=self.is_training
                    )
                ),
                rate=self.dropout,
            )  # batch_size x 512
            self.pi = Dense(s_fc2, self.action_size)  # batch_size x self.action_size
            self.prob = tf.nn.softmax(self.pi)
            self.v = Tanh(Dense(s_fc2, 1))  # batch_size x 1

            self.calculate_loss()

    def conv2d(self, x, out_channels, padding):
        return tf.layers.conv2d(x, out_channels, kernel_size=[3, 3], padding=padding)

    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi = tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(
            self.target_vs, tf.reshape(self.v, shape=[-1,])
        )
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)


def init_model():
    global session

    assert session is None

    with tf.Session() as temp_sess:
        temp_sess.run(tf.global_variables_initializer())
    session.run(tf.variables_initializer(model.graph.get_collection("variables")))

    return None


def train(model, examples):
    global session
    global epochs
    global batch_size
    global dropout

    batch_count = boards.shape[0] / batch_size

    for epoch in range(epochs):
        sample_ids = np.random.randint(boards.shape[0], size=batch_size)
        boards = examples[0][sample_ids]
        pis = examples[1][sample_ids]
        vs = examples[2][sample_ids]

        input_dict = {
            model.input_boards: boards,
            model.target_pis: pis,
            model.target_vs: vs,
            model.dropout: dropout,
            model.is_training: True,
        }

        pi_loss, v_loss = session.run(
            [model.loss_pi, model.loss_v], feed_dict=input_dict
        )


def predict(model, board):
    global session
    prob, v = session.run(
        [model.prob, model.v],
        feed_dict={
            model.input_boards: board,
            model.dropout: 0,
            model.is_training: False,
        },
    )
    return prob[0], v[0]


def save_checkpoint(model, model_id, checkpoint_path):
    filepath = os.join(checkpoint_path, "{}.pth.tar".format(model_id))

    global saver
    global session
    assert not (session is None)

    if saver is None:
        saver = tf.train.Saver(model.graph.get_collection("variables"))
    with model.graph.as_default():
        saver.save(session, filepath)


def load_checkpoint(model, model_id, checkpoint_path):
    filepath = os.join(checkpoint_path, "{}.pth.tar".format(model_id))
    if not os.path.exists(filepath + ".meta"):
        raise ("Cannot find model within {}".format(filepath))

    global saver
    global session
    assert not (saver is None)
    assert not (session is None)

    with model.graph.as_default():
        saver = tf.train.Saver()
        saver.restore(session, filepath)
