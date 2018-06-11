import tensorflow as tf


def build_embed(inputs, input_voca_size, output_size, scope='embedding', reuse=False, zero_pad=True):
    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        embed_tensor = tf.get_variable(name='embedding', shape=[input_voca_size, output_size], dtype=tf.float32)

        if zero_pad:
            zeros = tf.zeros(shape=[1, output_size], dtype=tf.float32)
            embed_tensor = tf.concat([zeros, embed_tensor[1:, :]], axis=0)

        enc = tf.nn.embedding_lookup(params=embed_tensor, ids=inputs)

        return enc


class RNNCell(object):
    def __init__(self, batch_size, hidden_layer_size, num_stack):
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.num_stack = num_stack

    def __call__(self, inputs, hidden_states):

        next_input = inputs

        for i in range(self.num_stack):
            next_input, cur_hidden_states = self.rnn_layer(inputs=next_input,
                                                           hidden_states=tf.split(value=hidden_states,
                                                                                  num_or_size_splits=self.num_stack,
                                                                                  axis=-1)[i],
                                                           scope="RNN_layer_{}".format(i+1),
                                                           reuse=False)

            if i == 0:
                next_hidden_states = cur_hidden_states
            else:
                next_hidden_states = tf.concat([next_hidden_states, cur_hidden_states], axis=-1)

        output = next_input

        return output, next_hidden_states

    def rnn_layer(self, inputs, hidden_states, scope, reuse=False):

        prev_cell_state, prev_hidden_state = tf.split(value=hidden_states, num_or_size_splits=2, axis=-1)

        with tf.variable_scope(name_or_scope=scope, reuse=reuse):
            next_hidden_state = tf.concat([prev_hidden_state, inputs], axis=-1)
            f = tf.layers.dense(next_hidden_state, self.hidden_layer_size, activation=tf.nn.sigmoid, name='forget')
            i = tf.layers.dense(next_hidden_state, self.hidden_layer_size, activation=tf.nn.sigmoid, name='input')
            o = tf.layers.dense(next_hidden_state, self.hidden_layer_size, activation=tf.nn.tanh, name='output')
            g = tf.layers.dense(next_hidden_state, self.hidden_layer_size, activation=tf.nn.sigmoid, name='gate')

            next_cell_state = tf.multiply(prev_cell_state, f)
            next_cell_state += tf.multiply(i, g)
            next_hidden_state = tf.multiply(o, tf.nn.tanh(next_cell_state))
            output = next_hidden_state
            next_hidden_state = tf.concat(values=[next_cell_state, next_hidden_state], axis=-1)

        return output, next_hidden_state

    def get_init_state(self):
        init_hidden_states = tf.zeros(shape=[self.batch_size, 2 * self.hidden_layer_size * self.num_stack])
        return init_hidden_states


def build_rnn(cell, inputs, max_len):

    init_hidden_state = cell.get_init_state()
    outputs = tf.TensorArray(dtype=tf.float32, size=max_len)
    loop_vars = [tf.convert_to_tensor(0, dtype=tf.int32), inputs, outputs, init_hidden_state]

    def _cond(i, *args):
        return tf.less(i, max_len)

    def _body(i, inputs, outputs, hidden_states):
        cur_input = inputs[:, i, :]
        cur_output, next_states = cell(inputs=cur_input, hidden_states=hidden_states)
        outputs = outputs.write(i, cur_output)
        i = i + 1
        return i, inputs, outputs, next_states

    _, _, outputs, last_state = tf.while_loop(cond=_cond, body=_body, loop_vars=loop_vars)
    outputs = tf.transpose(outputs.stack(), perm=[1, 0, 2])

    return last_state, outputs

















