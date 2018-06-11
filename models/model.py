from models.model_layers import *


class GenModel(object):
    def __init__(self, inputs, targets, batch_size, input_voca_size, embedding_size, layer_size, num_stack, max_len):

        self.inputs = inputs
        self.targets = targets

        self.batch_size = batch_size
        self.input_voca_size = input_voca_size
        self.embedding_size = embedding_size
        self.layer_size = layer_size
        self.num_stack = num_stack
        self.max_len = max_len
        self._build()

    def _build(self, inputs, targets):

        self.enc = build_embed(inputs=inputs, input_voca_size=self.input_voca_size)
        self.cell = RNNCell(batch_size=self.batch_size, hidden_layer_size=self.layer_size, num_stack=self.num_stack)
        last_state, outputs = build_rnn(cell=self.cell, inputs=self.enc, max_len=self.max_len)

        last_state = tf.split(last_state, self.num_stack, axis=-1)[-1]  # [ stacked state by # of num_stack ]
        last_state = tf.split(last_state, 2, axis=-1)[-1]  # [cell_state, hidden_state]

        self.logits = tf.layers.dense(inputs=last_state, units=self.input_voca_size, name='logits')
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.targets, logits=self.logits),
                                   axis=-1)
        self.accruacy = None

        return None












