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
  N=np.shape(X)[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(N):
            f= np.dot(X[i,:],W)
            const=-np.max(f)
            f_exp_norm=np.exp(f+const)/np.sum( np.exp(f+const) ) 
            for j in range(f_exp_norm.shape[0]):
                if j==y[i]:
                    loss+= -np.log(f_exp_norm[ y[i] ])
                    dW[:,j]+= X[i]*( f_exp_norm[j] -1 )
                else:
                    dW[:,j]+= X[i]*f_exp_norm[j] 

  loss/=N
  dW/=N
  loss+= (reg* np.sum(W**2) ) 
  dW+=2*reg*W
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
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores, axis=1)[:, np.newaxis]

  # FORWARD PATH
  # num = np.exp(scores[range(num_train),y])
  # den = np.sum(np.exp(scores), axis=1, keepdims=True)
  # probs = num / den
  # log_probs = -np.log(probs)
  # loss = np.sum(log_probs)

  # scores = X.dot(W)
  num = np.exp(scores[range(num_train), y])
  den = np.sum(np.exp(scores), axis=1)
  probs_true = num / den
  log_probs = -np.log(probs_true)
  loss = np.sum(log_probs)

  p = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  p[range(num_train), y] -= 1
  p /= num_train
  dW = X.T.dot(p)

  dW += reg * W
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

