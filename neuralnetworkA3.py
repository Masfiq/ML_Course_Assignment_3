
import numpy as np
import optimizers as opt


class NeuralNetwork():
    """
    A class that represents a neural network for nonlinear regression.
    """

    def __init__(self, n_inputs, n_hiddens_each_layer, n_outputs):
        """Creates a neural network with the given structure."""

        self.n_inputs = n_inputs
        self.n_hiddens_each_layer = n_hiddens_each_layer
        self.n_outputs = n_outputs

      
        shapes = []
        input_size = n_inputs
        for n_hidden in n_hiddens_each_layer:
            shapes.append((input_size + 1, n_hidden))  
            input_size = n_hidden
        shapes.append((input_size + 1, n_outputs))  

       
        self.all_weights, self.Ws = self._make_weights_and_views(shapes)

      
        self.all_gradients, self.Grads = self._make_weights_and_views(shapes)

        self.X_means = None
        self.X_stds = None
        self.T_means = None
        self.T_stds = None

        self.n_epochs = 0
        self.error_trace = []

    def _make_weights_and_views(self, shapes):
        """Creates vector of all weights and views for each layer."""
   
        total_weights = sum((rows * cols) for rows, cols in shapes)
        all_weights = np.random.uniform(-1, 1, total_weights)

      
        Ws = []
        start = 0
        for rows, cols in shapes:
            end = start + rows * cols
            W = np.reshape(all_weights[start:end], (rows, cols))
            W /= np.sqrt(rows)  # Divide by sqrt of number of inputs
            Ws.append(W)
            start = end

        return all_weights, Ws

    def __repr__(self):
        return f'NeuralNetwork({self.n_inputs}, {self.n_hiddens_each_layer}, {self.n_outputs})'

    def __str__(self):
        if self.n_epochs > 0:
            return f'{self.__repr__()} trained for {self.n_epochs} epochs with a final RMSE of {self.error_trace[-1]}'
        else:
            return f'{self.__repr__()} has not been trained.'

    def train(self, Xtrain, Ttrain, Xvalidate, Tvalidate, n_epochs, batch_size=-1,
              method='sgd', learning_rate=None, momentum=0, weight_penalty=0, verbose=True):
        """Updates the weights."""
        
        self.batch_size = batch_size
        
        # Standardize Xtrain, Ttrain, Xvalidate and Tvalidate
        self.X_means = Xtrain.mean(axis=0)
        self.X_stds = Xtrain.std(axis=0)
        Xtrain_standardized = (Xtrain - self.X_means) / self.X_stds
        Xvalidate_standardized = (Xvalidate - self.X_means) / self.X_stds
        
        self.T_means = Ttrain.mean(axis=0)
        self.T_stds = Ttrain.std(axis=0)
        Ttrain_standardized = (Ttrain - self.T_means) / self.T_stds
        Tvalidate_standardized = (Tvalidate - self.T_means) / self.T_stds

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = opt.Optimizers(self.all_weights)

        # Select optimization method
        if method == 'sgd':
            self.error_trace = optimizer.sgd(Xtrain_standardized, Ttrain_standardized,
                                             Xvalidate_standardized, Tvalidate_standardized,
                                             self.error_f, self.gradient_f,
                                             n_epochs=n_epochs, batch_size=batch_size,
                                             learning_rate=learning_rate,
                                             momentum=momentum, weight_penalty=weight_penalty,
                                             verbose=verbose)

        elif method == 'adam':
            self.error_trace = optimizer.adam(Xtrain_standardized, Ttrain_standardized,
                                              Xvalidate_standardized, Tvalidate_standardized,
                                              self.error_f, self.gradient_f,
                                              n_epochs=n_epochs, batch_size=batch_size,
                                              learning_rate=learning_rate,
                                              weight_penalty=weight_penalty, verbose=verbose)

        elif method == 'scg':
            self.error_trace = optimizer.scg(Xtrain_standardized, Ttrain_standardized,
                                             Xvalidate_standardized, Tvalidate_standardized,
                                             self.error_f, self.gradient_f,
                                             n_epochs=n_epochs, batch_size=batch_size,
                                             weight_penalty=weight_penalty, verbose=verbose)

        else:
            raise Exception("Method must be 'sgd', 'adam', or 'scg'")

        self.n_epochs += len(self.error_trace)
        self.best_epoch = optimizer.best_epoch

        return self

    def _add_ones(self, X):
        return np.insert(X, 0, 1, axis=1)

    def _forward(self, X):
        """Calculate outputs of each layer given inputs in X."""
        self.Zs = [self._add_ones(X)]
        for W in self.Ws[:-1]:
            Z = np.tanh(self.Zs[-1] @ W)
            self.Zs.append(self._add_ones(Z))
        self.Zs.append(self.Zs[-1] @ self.Ws[-1])
        return self.Zs

    def error_f(self, X, T):
        """Calculate mean squared error."""
        Y = self._forward(X)[-1]
        return np.mean((T - Y) ** 2)

    def gradient_f(self, X, T):
        """Return gradients with respect to all weights."""
        n_samples = X.shape[0]
        delta = -(T - self.Zs[-1]) / n_samples

        for layeri in range(len(self.Ws) - 1, -1, -1):
            self.Grads[layeri][:] = self.Zs[layeri].T @ delta
            if layeri > 0:
                delta = (delta @ self.Ws[layeri].T)[:, 1:] * (1 - self.Zs[layeri][:, 1:] ** 2)
                
        return self.all_gradients

    def use(self, X):
        """Return the output of the network for input samples."""
        X_standardized = (X - self.X_means) / self.X_stds
        Y = self._forward(X_standardized)[-1]
        return Y * self.T_stds + self.T_means

    def get_error_trace(self):
        """Returns list of root-mean square error for each epoch."""
        return self.error_trace
