import tensorflow as tf
import numpy as np

class Network(object):
    def __init__(self, input, params):                
        self.params = params
        self.vars = []
        self.vardict = {}
        self.batch_size = int(input.get_shape()[0])
        self.add_('input', input)
        self.setup()

    #"abstract" method :) 
    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def get_unique_name_(self, prefix):        
        id = sum(t.startswith(prefix) for t,_ in self.vars)+1
        return '%s_%d'%(prefix, id)

    def add_(self, name, var):
        self.vars.append((name, var))
        self.vardict[name] = var

    def get_output(self):
        return self.vars[-1][1]

    def conv(self, h, w, c_i, c_o, stride=1, name=None):
        name = name or self.get_unique_name_('conv')
        with tf.variable_scope(name) as scope:
            weights = self.params[name][0].astype(np.float32)
            conv = tf.nn.conv2d(self.get_output(), weights, [stride]*4, padding='SAME') # W*X
            if len(self.params[name]) > 1: # if biases are available
                biases = self.params[name][1].astype(np.float32) 
                bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list()) # conv + biases
                relu = tf.nn.relu(bias, name=scope.name) # relu(W*X + b)
            else:
                relu = tf.nn.relu(conv, name=scope.name)            
            self.add_(name, relu)
        return self

    def pool(self, size=2, stride=2, name=None):
        name = name or self.get_unique_name_('pool')
        # pool = tf.nn.avg_pool(self.get_output(),
        pool = tf.nn.max_pool(self.get_output(),
                              ksize=[1, size, size, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME',
                              name=name)
        self.add_(name, pool)
        return self

    def fc (self, name=None, withRelu=True):
        name = name or self.get_unique_name_('fc')
        with tf.variable_scope(name) as scope:
            
            input_x = self.get_output() #output from prev layer
            #shape = input_x.shape
            shape = input_x.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                 dim *= d
            x = tf.reshape(input_x, [-1, dim])
            
            weights = self.params[name][0].astype(np.float32)
            biases = self.params[name][1].astype(np.float32) 
            
            mult = tf.matmul(x, weights)
            biased = tf.nn.bias_add(mult, biases)
            if (withRelu):
                fc = tf.nn.relu(biased, name=scope.name)
            else:
                fc = biased # 
                
            self.add_(name, fc)
        return self
