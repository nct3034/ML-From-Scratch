import numpy as np
from models.base_model import BaseModel

class Node:
    """
    Represents a single decision point or leaf in the tree.
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree(BaseModel):
    """
    Decision Tree Classifier built from scratch.
    """
    def __init__(self, max_depth=10, min_samples_split=2):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_splits = min_samples_split
       
    def fit(self, x, y):
        """
        Public method to start training the tree.
        """
        # TODO: Call the recursive _grow_tree method and assign the result to self.root
        pass

    def predict(self, x):
        """
        Public method to predict labels for new data.
        """
        # TODO: Loop through each row in X, pass it to _traverse_tree, and return a numpy array of predictions
        pass

    def _grow_tree(self, x, y, depth=0):
        """
        Recursive function to build the tree branches.
        """
        # TODO: Step 1 - Check stopping criteria (max_depth reached, pure node, or too few samples)
        # If stopping criteria met, return a leaf Node containing the most common label in y
        
        # TODO: Step 2 - Find the best split using _best_split()
        
        # TODO: Step 3 - If a valid split is found, divide X and y into left and right datasets
        
        # TODO: Step 4 - Recursively call _grow_tree on the left and right datasets
        
        # TODO: Step 5 - Return a new Node containing the best feature, threshold, left_child, and right_child
        pass

    def _best_split(self, x, y):
        """
        Iterates through all features and values to find the split with the highest Information Gain.
        """
        # TODO: Loop through every column in X
        # TODO: Loop through every unique value in that column
        # TODO: Calculate information gain for each combination
        # TODO: Return the feature index and threshold that produced the best information gain
        pass

    def _split(self, x_column, split_thresh):
        """
        Helper method to divide data indices based on a threshold.
        """
        # TODO: Return the indices of rows where the column value matches the threshold (or is <= for numbers)
        # TODO: Return the indices of rows where it does not match
        pass

    def _entropy(self, y):
        """
        Calculates the impurity of a label array.
        """
        # Calculate the proportion of each class in y
        labels, counts = np.unique(y,return_counts=True)
        total_samples = len(y)
        proportions = counts / total_samples

        # Apply the entropy formula: -sum(p * log2(p))
        return -np.sum(proportions * np.log2(proportions))
        

    def _information_gain(self, y, x_column, split_thresh):
        """
        Calculates how much a split reduces entropy.
        """
        # TODO: Calculate parent entropy using _entropy(y)
        # TODO: Split the data using _split()
        # TODO: Calculate the weighted average entropy of the children
        # TODO: Return (parent_entropy - child_entropy)
        pass

    def _most_common_label(self, y):
        """
        Returns the most frequent label in a given array.
        """
        # TODO: Find and return the most common value in the array y
        pass

    def _traverse_tree(self, x, node):
        """
        Navigates a single sample down the tree to find its prediction.
        """
        # TODO: If the current node has a 'value' (is a leaf), return that value
        # TODO: Otherwise, check x against the node's feature_index and threshold
        # TODO: Recursively call _traverse_tree on the left or right child based on the check
        pass