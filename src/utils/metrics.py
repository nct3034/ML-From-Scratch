import numpy as np

class ClassificationMetrics:
    """
    A class to calculate various classification metrics based on true and predicted labels.
    """
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

        self.TP = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.TN = np.sum((self.y_true == 0) & (self.y_pred == 0))
        self.FP = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.FN = np.sum((self.y_true == 1) & (self.y_pred == 0))

    def accuracy(self):
        """
        (TP + TN) / Total
        """
        total = len(self.y_true)
        if total == 0:
            return 0.0
        return (self.TP + self.TN) / total

    def precision(self):
        """
        TP / (TP + FP)
        """
        if (self.TP + self.FP) == 0:
            return 0.0
        return self.TP / (self.TP + self.FP)

    def recall(self):
        """
        TP / (TP + FN)
        """
        if (self.TP + self.FN) == 0:
            return 0.0
        return self.TP / (self.TP + self.FN)

    def f1_score(self):
        """
        2 * (Precision * Recall) / (Precision + Recall)
        """
        precision = self.precision()
        recall = self.recall()
        if precision + recall == 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)