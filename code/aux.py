import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



class NeuralNetwork:
    """
    Class to define a Neural Network with feedforward fully connected layers

    """

    

    def __init__(self, sizes) -> None:
        """
        Sizes determines the size of the network. The first and last elements are input and outputs. The length must be at least 3.
        """
        assert type(sizes) == list
        assert len(sizes) > 2
        self._sizes = sizes

        self.weights = [np.random.randn(sizes[i], sizes[i - 1]) for i in range(1, len(sizes))]
        self.biases = [np.random.randn(sizes[i]) for i in range(1, len(sizes))]


    def eval(self, input):
        outputs = [np.zeros(s) for s in self._sizes]
        outputs[0] = input
        for i in range(len(outputs) - 1):
            outputs[i + 1] = sigmoid(self.weights[i] @ outputs[i] + self.biases[i])

        return outputs
    

    def cost(self, input, y): 
        outputs = self.eval(input)

        return np.sum((outputs[-1] - y)**2) / 2
    

    def cost_set(self, X, Y): 
        return np.mean([self.cost(x, y) for x, y in zip(X, Y)])

    def gradients(self, input, y):
        outputs = self.eval(input)
        der_sigmoid = [(1 - o) * o for o in outputs]

        delta = [np.random.randn(self._sizes[i], self._sizes[i - 1]) for i in range(1, len(self._sizes))]

        deltas = [outputs[-1] - y] 
        
        for i in reversed(range(len(self.weights) - 1)):
            delta = self.weights[i + 1].T @ deltas[-1] * der_sigmoid[i + 1]
            deltas.append(delta)

        deltas.reverse()

        der_weights = []
        der_biases = []

        for i in range(len(self.weights)):
            der_weights.append(np.outer(deltas[i], outputs[i].T))
            der_biases.append(np.sum(deltas[i], axis=0, keepdims=True))

        return der_weights, der_biases
    

    def train_adam(self, X, Y, max_it = 1e4):

        costs = []
        it = 0

        alpha_biases = 0.001
        alpha_weights = 0.001

        eps_biases = 1e-8
        eps_weights = 1e-8

        b1_biases = 0.9
        b1_weights = 0.9

        b2_biases = 0.999
        b2_weights = 0.999

        m_biases = [np.zeros_like(b) for b in self.biases]
        m_weights = [np.zeros_like(w) for w in self.weights]

        v_biases = [np.zeros_like(b) for b in self.biases]
        v_weights = [np.zeros_like(w) for w in self.weights]

        c = self.cost_set(X, Y)

        while it < max_it: 
            it += 1
            
            der_weights, der_biases = self.gradients(X[0], Y[0])
            
            # computing gradients
            for x, y in zip(X[1:], Y[1:]): 
                der_weights_, der_biases_ = self.gradients(x, y)
                der_biases = [d + d_ for (d, d_) in zip(der_biases, der_biases_)]
                der_weights = [d + d_ for (d, d_) in zip(der_weights, der_weights_)]

            der_biases = [d / len(X) for d in der_biases]
            der_weights = [d / len(X) for d in der_weights]
            
         
            m_biases = [b1_biases * m + (1 - b1_biases) * g for (m, g) in zip(m_biases, der_biases)]
            v_biases = [b2_biases * v + (1 - b2_biases) * g**2 for (v, g) in zip(v_biases, der_biases)]

            self.biases = [b - alpha_biases * m / (1 - b1_biases**it) / (eps_biases + np.sqrt(v / (1 - b2_biases**it))) for (b, m, v) in zip(self.biases, m_biases, v_biases)]

            m_weights = [b1_weights * m + (1 - b1_weights) * g for (m, g) in zip(m_weights, der_weights)]
            v_weights = [b2_weights * v + (1 - b2_weights) * g**2 for (v, g) in zip(v_weights, der_weights)]

            self.weights = [w - alpha_weights * m / (1 - b1_weights**it) / (eps_weights + np.sqrt(v / (1 - b2_weights**it))) for (w, m, v) in zip(self.weights, m_weights, v_weights)]

            c_ = self.cost_set(X, Y)
            
            if np.abs(c - c_) < 1e-5: 
                break
            else: 
                c = c_
                costs.append(c)

        return costs
            
    def accuracy(self, X, Y): 
        Y_candidate = np.zeros(len(Y))
        Y_true = np.zeros(len(Y))
        
        for i, x in enumerate(X): 
            Y_candidate[i] = np.argmax(self.eval(x)[-1])
            Y_true[i] = np.argmax(Y[i])

        return Y_true, Y_candidate


class SupportVectorMachine:
    def __init__(self, feature_map = None):
        self.weights = None
        self.biases = None
        if feature_map is None: 
            self.feature_map = lambda x: x
    
    def fit(self, X, y, epochs = int(1e4), learning_rate = 0.01, lambda_param = 1):
        X = self.feature_map(X)
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1

        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Gradient for correctly classified point
                    self.w -= learning_rate * (2 * lambda_param * self.w)
                else:
                    # Gradient for misclassified point
                    self.w -= learning_rate * (2 * lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= learning_rate * y[idx]

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        - X: Input data, shape (n_samples, n_features).

        Returns:
        - Predicted class labels, shape (n_samples,).
        """
        X = self.feature_map(X)
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)