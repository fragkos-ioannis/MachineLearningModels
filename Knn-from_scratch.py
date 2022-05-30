import numpy as np
from math import sqrt
from collections import Counter

def euclidian_distance(x1:list, x2:list):
    return sqrt(sum( (x1[i] - x2[i])**2 for i in range(len(x1)) )) 

class KNN:
    """Input: data of n dim,  k neighbors 
       Output: The class of the new feature."""
    def __init__(self, k=1) -> None:
       self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute the new feature's distances. 
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]

        # get k nearest samples and their labels.
        k_indices = np.argsort(distances)[:self.k] #argsort it's just the sort function
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote, the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0] # the most_common method ruturns [(number, frequency of the number)]
        # In order to select just the number, we use list[0][0]

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)