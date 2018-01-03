import numpy as np
import functools, operator

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params.setdefault('W1', np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale)
    self.params.setdefault('b1', np.zeros(num_filters))
    self.params.setdefault('W2', np.random.randn(num_filters * (H / 2) * (W / 2), hidden_dim) * weight_scale)
    self.params.setdefault('b2', np.zeros(hidden_dim))
    self.params.setdefault('W3', np.random.randn(hidden_dim, num_classes) * weight_scale)
    self.params.setdefault('b3', np.zeros(num_classes))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    scores, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    old_shape = scores.shape
    scores = scores.reshape((old_shape[0], 
                             functools.reduce(operator.mul, old_shape[1:], 1)))
    scores, cache2 = affine_relu_forward(scores, W2, b2)
    scores, cache3 = affine_forward(scores, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, grad = softmax_loss(scores, y)
    grad, dW, db = affine_backward(grad, cache3)
    grads.setdefault('W3', dW)
    grads.setdefault('b3', db)
    grad, dW, db = affine_relu_backward(grad, cache2)
    grads.setdefault('W2', dW)
    grads.setdefault('b2', db)
    grad = grad.reshape(old_shape)
    grad, dW, db = conv_relu_pool_backward(grad, cache1)
    grads.setdefault('W1', dW)
    grads.setdefault('b1', db)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


class MYConvNet(object):
  def _get_size_conv(self, w, ks, H, W, stride):
    F, _, HH, WW = w.shape
    pad = (ks - 1) / 2
    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride
    return H_out, W_out
  def _get_size_pool(self, H, W, HH, WW, stride):
    H_out = (H - HH) / stride + 1
    W_out = (W - WW) / stride + 1
    return H_out, W_out
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 32], filter_size=[5, 5],
               hidden_dim=(512, 256, 10,), use_pooling=[True, True], weight_scale=[1e-4, 1e-3, 1e-3, 1e-2, 1e-2], reg=0.0,
               dtype=np.float32):
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    self.num_filters = num_filters
    self.filter_size = filter_size
    self.hidden_dim = hidden_dim
    self.use_pooling = use_pooling
    
    C, H, W = input_dim
    
    index = 0
    for number, size, pooling in zip(num_filters, filter_size, use_pooling):
      Wi = 'W{}'.format(index)
      bi = 'b{}'.format(index)
      self.params.setdefault(Wi, np.random.randn(number, C, size, size) * weight_scale[index])
      self.params.setdefault(bi, np.zeros(number))
      H, W = self._get_size_conv(self.params[Wi], size, H, W, 1)
      if pooling:
        H, W = self._get_size_pool(H, W, 2, 2, 2)
      C = number
      index += 1
    
    pre_hidden_dim = C * H * W
    for number in hidden_dim:
      Wi = 'W{}'.format(index)
      bi = 'b{}'.format(index)
      self.params.setdefault(Wi, np.random.randn(pre_hidden_dim, number) * weight_scale[index])
      self.params.setdefault(bi, np.zeros(number))
      pre_hidden_dim = number
      index += 1

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    scores = X
    caches = []
    index = 0
    for pooling in self.use_pooling:
      W, b = self.params['W{}'.format(index)], self.params['b{}'.format(index)]
      if pooling:
        scores, cache = conv_relu_pool_forward(
            scores, W, b, 
            conv_param={'stride': 1, 'pad': (W.shape[2] - 1) / 2}, 
            pool_param={'pool_height': 2, 'pool_width': 2, 'stride': 2})
        caches.append((cache, conv_relu_pool_backward))
      else:
        scores, cache = conv_relu_forward(
            scores, W, b,
            conv_param={'stride': 1, 'pad': (W.shape[2] - 1) / 2})
        caches.append((cache, conv_relu_backward))
      index += 1
    
    old_shape = scores.shape
    scores = scores.reshape((old_shape[0],
                             functools.reduce(operator.mul, old_shape[1:], 1)))
    caches.append((old_shape, 'reshape'))
    
    for hidden_index, _ in enumerate(self.hidden_dim):
      W, b = self.params['W{}'.format(index)], self.params['b{}'.format(index)]
      if hidden_index + 1 < len(self.hidden_dim):
        scores, cache = affine_relu_forward(scores, W, b)
        caches.append((cache, affine_relu_backward))
      else:
        scores, cache = affine_forward(scores, W, b)
        caches.append((cache, affine_backward))
      index += 1
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss, grad = softmax_loss(scores, y)
    for cache, layer in reversed(caches):
      if layer != 'reshape':
        index -= 1
        grad, dW, db = layer(grad, cache)
        grads.setdefault('W{}'.format(index), dW)
        grads.setdefault('b{}'.format(index), db)
      else:
        grad = grad.reshape(cache)
    return loss, grads
