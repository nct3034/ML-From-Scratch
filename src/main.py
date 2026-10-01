import argparse
import os
import numpy as np
from utils.data_loader import load_data
from preprocessing.encoders import LabelEncoder
from utils.metrics import ClassificationMetrics
from models.decision_tree import DecisionTree

def main():
    parser = argparse.ArgumentParser(description="Run ML Models from Scratch")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to run (e.g., decision_tree)")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset (e.g., data/play_tennis.csv)")
    parser.add_argument("--compare", action="store_true", help="Run Scikit-Learn model to compare results")
    parser.add_argument("--debug", action="store_true", help="Log detailed calculations to a file")

    args = parser.parse_args()
    print("========================================")
    print(f" Initializing Model: {args.model}")
    print(f" Loading Dataset: {args.data}")
    print("========================================\n")

    x, y = load_data(args.data)
    if x is None or y is None:
        print("Execution aborted due to data loading error.")
        return

    # At this point, you would add your LabelEncoder loop for x and y,
    # then instantiate your selected model (like Decision Tree) based on args.model,
    # and finally grade it using ClassificationMetrics.

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    tree = DecisionTree()
    dataset_entropy = tree._entropy(y_encoded)
    print(f"Entropy of dataset: {dataset_entropy}")

if __name__ == "__main__":
    main()