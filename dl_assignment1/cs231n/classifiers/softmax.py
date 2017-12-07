import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, _ = X.shape
  D, C = W.shape
  for i in range(N):
    value = 0.0
    total_expsum = 0.0
    
    tmp = np.zeros(C)
    for j in range(C):
      for k in range(D):
        tmp[j] += X[i, k] * W[k, j]
      total_expsum += np.exp(tmp[j])
    
    loss += -tmp[y[i]] + np.log(total_expsum)
    
    for k in range(D):
      dW[k, y[i]] -= X[i, k]
    for j in range(C):
      for k in range(D):
        dW[k, j] += np.exp(tmp[j]) / total_expsum * X[i, k]
        
  loss /= N
  dW /= N
  for i in range(D):
    for j in range(C):
      loss += reg * W[i, j] * W[i, j]
      dW[i, j] += 2 * reg * W[i, j]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, _ = X.shape
  D, C = W.shape
  
  value = np.matmul(X, W) # N x C
  exp_val = np.exp(value) # N x C
  exp_sum = np.sum(exp_val, axis=1) # N x 1
  
  loss += np.mean(-value[np.arange(N), y] + np.log(exp_sum))
  loss += reg * np.sum(np.square(W))
  
  dW += np.matmul(X.T, (exp_val.T / exp_sum).T - np.eye(C)[y]) / N
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW

def softmax_loss_tensorflow(W, X, y, reg):
  import tensorflow as tf
  _, C = W.shape
  _W = tf.placeholder(tf.float64, W.shape)
  _X = tf.placeholder(tf.float64, X.shape)
  _y = tf.placeholder(tf.int64, y.shape)
  _reg = tf.placeholder(tf.float64)
  _loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(_y, C), 
                                                  logits=tf.matmul(_X, _W))
  _loss = tf.reduce_mean(_loss)
  _loss += reg * tf.reduce_sum(tf.square(_W))
  _dW = tf.gradients(_loss, _W)
  
  with tf.Session() as sess:
    loss, dW = sess.run([_loss, _dW], feed_dict={
      _W: W,
      _X: X,
      _y: y,
      _reg: reg
    })
  return loss, dW[0]