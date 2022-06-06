import numpy as np


class LinearRegression:
    """Input: Data
       Output: Return a value for forecasting or classification."""
    def __init__(self, lr=0.01, n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None 
        self.bias = None

    def fit(self, X, y):
        # initialize parameters
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            #Calculate the derivative of w and b 
            y_pred = np.dot(X, self.weight) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            #Update the weights and bias
            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weight) + self.bias
        return y_pred

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt 

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)

    def mse(y_pred, y):
        return np.mean((y - y_pred)**2)

    print(mse(predicted, y_test))

    y_pred_line = regressor.predict(X)    
    fig = plt.figure(figsize=(8,6))
    m1 = plt.scatter(X_train, y_train)
    m1 = plt.scatter(X_test, y_test)
    plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.show()
