import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.net = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
		
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        _X = X.copy()
        for layer in self.net:
            if isinstance(layer, FullyConnectedLayer):
                layer.W.grad = 0
                layer.B.grad = 0
            _X = layer.forward(_X)

        loss, grads = softmax_with_cross_entropy(_X, y)
        d_out = grads.copy()

        for layer in reversed(self.net):
            d_out = layer.backward(d_out)
		
        for layer in self.net:
            if isinstance(layer, FullyConnectedLayer):
                l2_loss, l2_grad = l2_regularization(layer.W.value, self.reg)
                loss += l2_loss
                layer.W.grad += l2_grad

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        _X = X.copy()
        pred = np.zeros(X.shape[0], np.int)
        for layer in self.net:
            _X = layer.forward(_X)
        probs = softmax(_X)
        pred = np.argmax(probs, axis=1)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for num_layer, layer in enumerate(self.net):
            for param_name, param in layer.params().items():
                result[(num_layer,param_name)] = param        

        return result
