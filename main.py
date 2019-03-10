# Digit Recognizer kaggle competition main file

# Imports
import numpy as np
import pandas as pd
import sys # For adding 

# Major function calls
digits_dataset = pd.read_csv('train.csv', sep=",")
print(digits_dataset.describe())
