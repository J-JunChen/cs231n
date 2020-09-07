from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    num_class = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        score = X[i].dot(W)
        score -= max(score)   # numeric stability
        loss_i = - score[y[i]] + np.log(np.sum(np.exp(score)))
        loss += loss_i
        for j in range(num_class):
            softmax_output = np.exp(score[y[j]]) / np.sum(np.exp(score))
            if j == y[i]:
                dW[:, j] += (-1 + softmax_output) * X[i]
            else:
                dW[:, j] += softmax_output * X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)  # L2 regularization
    dW = dW/num_train + reg * W  # derivative regularization
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    num_class = W.shape[1]
    num_train = X.shape[0]

    score = X.dot(W)
    score -= np.max(score, axis=1).reshape(-1, 1) # shape (500, 10)
    softmax_output = np.exp(score) / np.sum(np.exp(score), axis=1).reshape(-1,1)
    # print(softmax_output.shape)
    # print(range(num_train))
    # print(softmax_output[range(num_train), list(y)])
    # print(softmax_output[0, list(y)])
    loss = - np.sum(np.log(softmax_output[range(num_train), list(y)])) # one-hot coding: only wheni==j, yi==1;
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)

    dS = softmax_output.copy()
    dS[range(num_train), list(y)] += -1  # when i==j, dS += 1; when i!=j, dS = dS
    dW = (X.T).dot(dS)  # X.T.shape (D,N); X.shape (N,D); W.shape==dW.shape (D,C) 
    dW /= num_train 
    dW += reg * W
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
