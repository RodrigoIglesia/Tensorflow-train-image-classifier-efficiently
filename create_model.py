'''
Create simple models for image classification tasks.
Author: Rodrigo de la Iglesia.
Version: 1.0.
27/07/2019
'''
import tensorflow as tf 

def convLayer(input_data, filter_shape, num_filters, strides, padding, name, bias_relu=True):    
    with tf.variable_scope(name):
        num_input_channels = input_data.get_shape()[-1].value
        conv_filt_shape = [filter_shape, filter_shape, num_input_channels, num_filters]

        weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name='W', use_resource=True)
        bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name=name+'_b', use_resource=True) 
        
        out_layer = tf.nn.conv2d(input_data, weights, [1, strides, strides, 1], padding=padding, name='out')
        
        if bias_relu:
            out_layer += bias
            out_layer = tf.nn.relu(out_layer, name='relu')

        return out_layer

def poolLayer(input_data, pool_shape, strides, padding, name, avg=True):
    with tf.variable_scope(name):
        ksize = [1, pool_shape, pool_shape, 1]
        strides = [1, strides, strides, 1]
        
        if avg:
            out_layer = tf.nn.avg_pool(input_data, ksize=ksize, strides=strides, padding=padding, name='avg')
            
        out_layer = tf.nn.max_pool(input_data, ksize=ksize, strides=strides, padding=padding, name='max')
    
        return out_layer

def denseLayer(input_data, n_output, name, relu=True):
    with tf.variable_scope(name):
        n_input = input_data.get_shape()[-1].value
        weight_shape = [n_input, n_output]

        weights = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.03), name='W', use_resource=True)
        bias = tf.Variable(tf.truncated_normal([n_output]), name='b', use_resource=True)
    
        out_dense = tf.matmul(input_data, weights)
        out_dense = tf.add(out_dense, bias, name='out')
        if relu:
            out_dense = tf.nn.relu(out_dense, name='relu')

        return out_dense

def dropoutLayer(input_data, keep_prob):
    out_dropout = tf.nn.dropout(input_data, keep_prob)
    
    return out_dropout

def flattenLayer(input_data):
    layer_shape = input_data.get_shape()
  
    n_input = layer_shape[1:4].num_elements()
    flat_layer = tf.reshape(input_data,[-1,n_input])
  
    return flat_layer

def inceptionLayer(input_data, out_filters, intermed_filters, name):
    with tf.variable_scope(name):
        # conv. 1x1
        conv_1 = convLayer(input_data, 1, out_filters[0], 1, 'SAME', name='conv1x1', bias_relu=True)
        # conv. 1x1 prev. to 3x3
        conv_1_3 = convLayer(input_data, 1, intermed_filters[0], 1, 'SAME', name='conv1x1_prev3x3', bias_relu=True)
        # conv. 1x1 prev. to 5x5
        conv_1_5 = convLayer(input_data, 1, intermed_filters[1], 1, 'SAME', name='conv1x1_prev5x5', bias_relu=True)
        # conv. 3x3
        conv_3 = convLayer(conv_1_3, 3, out_filters[1], 1, 'SAME', name='conv3x3', bias_relu=True)
        # conv. 5x5
        conv_5 = convLayer(conv_1_5, 5, out_filters[2], 1, 'SAME', name='conv5x5', bias_relu=True)
        
        pooling = poolLayer(input_data, 3, 1, 'SAME', name='pool', avg=False)
        #conv. after pooling.
        conv_pool = convLayer(pooling, 1, out_filters[3], 1, 'SAME', name='conv_pool', bias_relu=True)
            
        out_inception = tf.concat([conv_1, conv_3, conv_5, conv_pool], axis=3, name='concat')  

        return out_inception

def lossFunc(logits, labels, name):
    with tf.variable_scope(name):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        
        return loss


class Model:
    '''
    Create tensorflow graph operations.
    '''
    def __init__(self, n_classes, data_x, data_y, lr):
        #Class number.
        self.n_classes = n_classes
        #Learning rate.
        self.lr = lr
        #Dropout value.
        self.keep_prob = tf.placeholder("float")
        self.makePred(data_x, data_y)
            
    def makePred(self, data_x, data_y):
        #Convolution 1.
        self.conv1 = convLayer(data_x, 3, 4, 2, 'VALID', name='conv_1', bias_relu=True)
        #Convolution 2.
        self.conv2 = convLayer(self.conv1, 3, 8, 1, 'VALID', name='conv_2', bias_relu=True)
        #Pooling 1.
        self.pool1 = poolLayer(self.conv2, 3, 2, 'SAME', name='pool_1', avg=False)
        #Convolution 3.
        self.conv3 = convLayer(self.pool1, 3, 8, 2, 'VALID', name='conv_3', bias_relu=True)
        #Convolution 4.
        self.conv4 = convLayer(self.conv3, 3, 16, 1, 'VALID', name='conv_4', bias_relu=True)
        #Pooling 2.
        self.pool2 = poolLayer(self.conv4, 3, 2, 'SAME', name='pool_2', avg=False)
        #Flat layer.
        flatten = flattenLayer(self.pool2)
        #Classification step.
        self.logits = denseLayer(flatten, self.n_classes, name='logits_layer', relu=False)
        self.y_pred = tf.nn.softmax(self.logits, name='predictions')
        
        #Loss function.
        self.loss = lossFunc(self.logits, data_y, name='loss_function')
        #Gradient descent optimizer.
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        y_cls = tf.argmax(data_y, 1, name='class')
        y_pred_cls = tf.argmax(self.y_pred, 1, name='class_predicted')
        correct = tf.equal(y_pred_cls, y_cls, name='correct_bool')
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')