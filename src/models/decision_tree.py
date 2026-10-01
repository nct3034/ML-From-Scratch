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
        # Step 1 - Check stopping criteria (max_depth reached, pure node, or too few samples)
        # If stopping criteria met, return a leaf Node containing the most common label in y
        if depth >= self.max_depth or len(y) < self.min_samples_splits or (len(np.unique(y)) == 1):
            leaf_value = self._most_common_label(y)
            return Node(value= leaf_value)

        # Step 2 - Find the best split using _best_split()
        feature_idx, thresh = self._best_split(x, y)
        
        # Step 3 - If a valid split is found, divide x and y into left and right datasets
        if feature_idx is not None and thresh is not None:
            x_column = x[:, feature_idx]
            left_idx, right_idx = self._split(x_column, thresh)

            x_left, y_left = x[left_idx, :], y[left_idx]
            x_right, y_right = x[right_idx, :], y[right_idx]

            # Step 4 - Recursively call _grow_tree on the left and right datasets
            left = self._grow_tree(x_left, y_left, depth + 1)
            right = self._grow_tree(x_right, y_right, depth + 1)

            # Step 5 - Return a new Node containing the best feature, threshold, left_child, and right_child
            return Node(feature_index=feature_idx, threshold=thresh, left=left, right=right)

        # Fallback: return Node if cannot split
        leaf_value = self._most_common_label(y)
        return Node(value=leaf_value)
        

    def _best_split(self, x, y):
        """
        Iterates through all features and values to find the split with the highest Information Gain.
        """
        best_ig = -1
        best_feature_idx = None
        best_thresh = None

        n_feature = x.shape[1]
        # Loop through every column in x
        for col_idx in range(n_feature):
            x_column = x[: col_idx]

            # Loop through every unique value in that column
            thresholds = np.unique(x_column)
            for threshold in thresholds:
                # Calculate information gain for each combination
                ig = self._information_gain(y, x_column, threshold)
                if ig > best_ig:
                    best_ig = ig
                    best_feature_idx = col_idx
                    best_thresh = threshold
        
        # Return the feature index and threshold that produced the best information gain
        return best_feature_idx, best_thresh
        


    def _split(self, x_column, split_thresh):
        """
        Helper method to divide data indices based on a threshold.
        """
        # Return the indices of rows where the column value matches the threshold (or is <= for numbers)
        left_idxs = np.argwhere(x_column <= split_thresh).flatten()
        # Return the indices of rows where it does not match
        right_idxs = np.argwhere(x_column > split_thresh).flatten()

        return left_idxs, right_idxs
        

    def _entropy(self, y):
        """
        Calculates the impurity of a label array.
        """
        # Calculate the proportion of each class in y
        labels, counts = np.unique(y,return_counts=True)
        total_samples = len(y)
        p = counts / total_samples

        # Apply the entropy formula: -sum(p * log2(p))
        return -np.sum(p * np.log2(p))
        

    def _information_gain(self, y, x_column, split_thresh):
        """
        Calculates how much a split reduces entropy.
        """
        # Calculate parent entropy using _entropy(y)
        parent_entropy = self._entropy(y)
        
        # Split the data using _split()
        left_idx, right_idx = self._split(x_column, split_thresh)

        # Calculate the weighted average entropy of the children
        y_left = y[left_idx]
        y_right = y[right_idx]

        if len(y_left) == 0 or len(y_right) == 0:
            return 0

        n_total = len(y)
        weight_left = len(y_left) / n_total
        weight_right = len(y_right) / n_total

        # Return (parent_entropy - child_entropy)
        child_entropy = (weight_left * self._entropy(y_left)) + (weight_right * self._entropy(y_right))
        information_gain = parent_entropy - child_entropy
        return information_gain
        

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