from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract Base Class for all machine learning models in this project.
    It forces every child class to implement fit() and predict() methods.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the feature matrix X and label vector y.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict labels for the given feature matrix X.
        """
        pass