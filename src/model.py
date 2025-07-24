import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=1.11, n_iterations=10000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
        
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            cost= 1/(2 * n_samples) * np.sum((y_predicted - y)**2)
            self.cost_history.append(cost)
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            # print(f"Iteration {i + 1}/{self.n_iterations}, Cost: {cost} weights: {self.weights}, bias: {self.bias}")
            # Check for convergence
            if len(self.cost_history) > 1 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print("Convergence reached.")
                self.stopped_at = i + 1
                break
            

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias  
    
    def score(self, X, y, predictions):
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
  
    
