import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    orig_preds = predictions.copy()
    if predictions.ndim > 1:
        orig_preds -= np.max(orig_preds, axis=1).reshape(orig_preds.shape[0], -1)
        probs = np.exp(orig_preds) / np.sum(np.exp(orig_preds), axis=1).reshape(orig_preds.shape[0], -1)
    else:
        orig_preds -= np.max(orig_preds)
        probs = np.exp(orig_preds) / np.sum(np.exp(orig_preds))
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
	
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if probs.ndim > 1:
        indexes = tuple(np.vstack((np.arange(target_index.shape[0]), target_index.reshape(1, -1))))
        cross_entropy = np.sum(-np.log(probs[indexes]))
    else:
        cross_entropy = -np.log(probs[target_index])
    return cross_entropy

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    a = np.zeros_like(predictions.copy(), dtype=np.float32)
    if predictions.ndim > 1:
        indexes = tuple(np.vstack((np.arange(target_index.shape[0]), target_index.reshape(1, -1))))
        a[indexes] = 1
        dprediction = softmax(predictions.copy()) - a
    else:
        a[target_index] = 1
        dprediction = softmax(predictions.copy()) - a
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.indexes = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.indexes = (X > 0)
        f_result = X.copy()
        f_result[~self.indexes] = 0
        return f_result
		

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out.copy()
        d_result[~self.indexes] = 0
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        self.W.grad = self.X.T.dot(d_out)
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)
        d_result = d_out.dot(self.W.value.T)
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
