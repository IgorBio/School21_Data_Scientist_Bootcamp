import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
import pandas as pd

# with zipfile.ZipFile('../datasets/client_info.csv.zip', 'r') as zip_ref:
#     zip_ref.extractall('../datasets')

df = pd.read_csv('../datasets/client_info.csv')

df.set_index("ID", inplace=True)
# df.head(5)


y = df["TARGET"]
x = df.drop("TARGET", axis=1)
# y.shape
# x.shape

X_cat = x.select_dtypes('object')
X_num = x.select_dtypes(('int64', 'float64'))
X_cat.shape[1]
X_num.shape[1]

