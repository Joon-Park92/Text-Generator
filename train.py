import tensorflow as tf

class Trainer(object):
    def __init__(self):
        self.train_writer = tf.summary.FileWriter()
        self.dev_writer = tf.summary.FileWriter()
        