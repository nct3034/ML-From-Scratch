import numpy as np

class LabelEncoder:
    """
    Translates categorical text data into integer values.
    """
    def __init__(self):
        self.mapping = {}
        self.inverse_mapping = {}

    def fit(self, data):
        """
        Learns the vocabulary. Finds all unique values in the 1D data array 
        and assigns a unique integer to each.
        """
        unique_values = np.unique(data)
        for integer_id, text_value in enumerate(unique_values):
            self.mapping[text_value] = integer_id
            self.inverse_mapping[integer_id] = text_value

    def transform(self, data):
        """
        Translates the data array using the learned vocabulary.
        """
        encoded_data = [self.mapping[val] for val in data]
        return np.array(encoded_data)

    def fit_transform(self, data):
        """
        A convenience method that learns the vocabulary and translates in one step.
        """
        self.fit(data)
        return self.transform(data)