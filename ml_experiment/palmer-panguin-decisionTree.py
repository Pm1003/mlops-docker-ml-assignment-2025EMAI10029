import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml

# Loading the Palmer Penguin Dataset
penguins = fetch_openml(name='penguins', version=1, as_frame=True)
df = penguins.frame

# Dropping the rows with missing target
df = df.dropna(subset=['species'])

# Separating the features and target
X = df.drop('species', axis=1)
y = df['species']

# Encoding the categorical columns
X = pd.get_dummies(X, drop_first=True)

# Encoding the target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Handling the missing values (now all numeric)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Training the Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Metrics
precision = precision_score(y_test, y_pred, average='macro')
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

print("Model Precision:", precision)
print("Model AUC Score:", auc)