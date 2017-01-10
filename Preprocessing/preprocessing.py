import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

dataset = pd.read_csv("Preprocessing/data.csv")
X = dataset.iloc[:,0:3]
Y = dataset.iloc[:, 3]

# Preprocessing
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X.iloc[:,1:3])

X.iloc[:,1:3] = imputer.transform(X.iloc[:,1:3])


## Categorization and one hot encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
### Countries
le = LabelEncoder()
le.fit(dataset.iloc[:,0])
X.iloc[:,0] = le.transform(dataset.iloc[:,0])

ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()

### Yes/No
le_y = LabelEncoder()
Y = le_y.fit_transform(Y)