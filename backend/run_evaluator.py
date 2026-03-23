import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.evaluator import evaluate_dataset

print("Starting evaluator...")
evaluate_dataset(
    dataset_path="C:/Projects/colony-counter/dataset/CEMTimages",
    max_images=200
)
print("Done.")