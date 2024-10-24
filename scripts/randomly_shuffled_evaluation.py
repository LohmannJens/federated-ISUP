import os
import sys

import numpy as np
import pandas as pd

from tensorflow import math
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

sys.path.append("..")
from training_evaluation.evaluation_helpers import plot_confusion_matrix


# define how often to run the test
n_runs = 10

results_dir = "randomly_shuffled_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created folder: {results_dir}")
else:
    print(f"Folder already exists: {results_dir}")

# define a file to save the stdout
file_path = os.path.join(results_dir, f"out.txt")

# path where to find the data, in this case the testdata is used --> needs to be changed by user
test_data = pd.read_csv(os.path.join("..", "testdata", "data", "client1", "test.csv"))
for id in range(n_runs):
    test_data['isup_shuffled'] = test_data['isup'].sample(frac=1).reset_index(drop=True)
    with open(file_path, "a") as file:
        sys.stdout = file
        print()
        print(f"run: {id}")
        print("#########")        
        acc = accuracy_score(test_data['isup'], test_data['isup_shuffled'])
        print("accuracy: ", acc)

        kappa = cohen_kappa_score(test_data['isup'], test_data['isup_shuffled'], weights='quadratic')
        print("kappa: ", kappa)
        
        f1 = f1_score(test_data['isup'], test_data['isup_shuffled'], average='macro')
        print("F1 score: ", f1)
    sys.stdout = sys.__stdout__

    # CONFUSION MATRIX
    mat = math.confusion_matrix(test_data['isup'], test_data['isup_shuffled'], num_classes=6)
    f = plot_confusion_matrix(np.array(mat, dtype='float32'), "isup", True)
    f.savefig(os.path.join(results_dir, f"{id}_confusion_matrix_relative.png"))
    f = plot_confusion_matrix(np.array(mat), "isup", False)
    f.savefig(os.path.join(results_dir, f"{id}_confusion_matrix.png"))

