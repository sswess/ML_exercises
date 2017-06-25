import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores_exp = np.exp(scores-np.max(scores, axis=1, keepdims=True))

  sum = np.sum(scores_exp, axis=1, keepdims=True)
  probability = scores_exp/sum
  #list containing the correct classification
  indices = [range(num_train), y]
  correct_class_score = probability[indices]

  #calculate -log(prob_y) and take the sum across all training examples
  loss = np.sum(-np.log(correct_class_score))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  #Compute Gradient
  probability[indices] -=1
  dW = X.T.dot(probability)
  dW /= num_train
  dW += .5 * reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW



def softmax_loss_vectorized(W, X, y, reg):
  #I solved the vectorized version by accident in the first part I'm going to optimize that code as best as I can

  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]


  scores = X.dot(W)
  scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))

  sum = np.sum(scores_exp, axis=1, keepdims=True)
  probability = scores_exp / sum
  indices = [range(num_train), y]
  correct_class_score = probability[indices]

  loss = np.sum(-np.log(correct_class_score))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  # Compute Gradient
  probability[indices] -= 1
  dW = X.T.dot(probability)
  dW /= num_train
  dW += .5 * reg * W

  return loss, dW

