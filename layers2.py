from    inits import *
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers
# from    config import args


def sparse_dropout(x, rate, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./(1 - rate))


def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs dense).
    返回x*y矩阵相乘
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Dense(layers.Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(layers.Layer):
    """
    Graph convolution layer.
    input_dim：上一层输出的第1维的长度，即提取后的特征长度h_l
    output_dim：实际上不是output的dimension，而是本层神经元的个数，即h_(l+1)
    """
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False, #
                 activation=tf.nn.relu,
                 bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        # super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_features_nonzero = num_features_nonzero

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight',shape=[self.input_dim, self.output_dim])
    #     self.weights_ = []
    #     for i in range(1): #这里只设置了一层，add_variable函数可能需要被调换，或者另写一个build函数
    #         w = self.add_variable('weight' + str(i), [input_dim, output_dim])
    #         self.weights_.append(w)
        if self.bias:
            self.bias = self.add_weight(name='bias', shape = [self.output_dim])

    def call(self, inputs, training=None, flatten=None):
        '''inputs格式：(features, support),features为H_l，support为(D^-0.5AD^0.5)
        training默认取None
        输出：H_(l+1)，维度为feature的长度（即d）'''
        x, support_ = inputs[:,:-28], inputs[:,-28:]

        if self.is_sparse_inputs:
            self.num_features_nonzero = x.values.shape

        # dropout，training取False且self.dropout取0时，无dropout(但training取默认None时有dropout，None不等于False)。想要有，要同时传入这两个参数
        if training is not False and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif training is not False:
            x = tf.nn.dropout(x, self.dropout)

        # convolve
        supports = list()
        if not self.featureless:
            pre_sup = dot(x, self.weight, sparse=self.is_sparse_inputs) # H*W
        else:
            pre_sup = self.weight

        # TODO：之后把sparse=True，只需把support_转为稀疏矩阵即可
        support = dot(support_, pre_sup, sparse=False)  # (D^-0.5)A(D^0.5)HW
        supports.append((support))
        # for i in range(len(support_)):
        #     if not self.featureless: # if it has features x
        #         pre_sup = dot(x, self.weights_[i], sparse=self.is_sparse_inputs)  # 即为H*W
        #     else:
        #         pre_sup = self.weights_[i]
        #
        #     support = dot(support_[i], pre_sup, sparse=True)   # 即为(D^-0.5)A(D^0.5)HW
        #     supports.append(support)
        output = tf.add_n(supports) #supports这个列表里，所有support相加（对应位置求和）

        # bias
        if self.bias:
            output += self.bias

        # if flatten:
        #     return self.activation(output).reshape()
        return self.activation(output)
