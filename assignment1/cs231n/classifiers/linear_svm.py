import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    diff_count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = (scores[j] - correct_class_score) + 1  # note delta = 1
      if margin > 0:
        diff_count += 1
        dW[:, j] += X[i]  # gradient update for incorrect rows
        loss += margin
    # gradient update for correct row
    dW[:, y[i]] -= (diff_count * X[i])

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg * W  # regularize the weights
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized1(W, X, y, reg,delta=1):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)

  correct_indices = np.array([range(num_train),y])
  correct_class_score = scores[correct_indices[0], correct_indices[1]]

  margins = np.maximum(0, scores.T - correct_class_score + delta)
  margins = margins.T
  margins[correct_indices[0], correct_indices[1]] = 0

  loss += np.sum(margins)
  loss /= num_train
  loss+=reg*np.sum(W*W)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  counts = (margins > 0).astype(int)
  counts[correct_indices[0], correct_indices[1]] = -np.sum(counts, axis=1)

  dW = X.T.dot(counts)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

def svm_loss_vectorized(W, X, y, reg):

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train=X.shape[0]



  margins = np.maximum(0, X.dot(W)-X.dot(W)[np.arange(num_train),y].reshape(-1,1)+1)
  margins[np.arange(num_train),y] = 0
  loss = np.sum(margins)

  # Compute the average
  loss /= num_train

  # Add regularization
  loss += reg*np.sum(W*W)



  margins[margins>0] = 1
  margins[np.arange(num_train),y] = -np.sum(margins,axis=1)
  dW = X.T.dot(margins)
  # Divide by the number of training examples
  dW /= num_train
  # Add regularization
  dW += reg*W



  return loss, dW