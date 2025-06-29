import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load any dataset
df = pd.read_csv("Churn_Modelling.csv")

df = df[0:100]


label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save the encoder


# Separate features and target
x = df.drop("IsActiveMember", axis=1)
y = df["IsActiveMember"]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initial model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(x_train, y_train)
y_pred = dt_classifier.predict(x_test)
print("Initial Accuracy:", accuracy_score(y_test, y_pred))

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
grid_search.fit(x_train, y_train)

best_dt_classifier = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate tuned model
y_pred_tuned = best_dt_classifier.predict(x_test)
print("Tuned Accuracy:", accuracy_score(y_test, y_pred_tuned))

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(best_dt_classifier, feature_names=x.columns, class_names=np.unique(y).astype(str), filled=True)
plt.show()

# Predict on new sample
# Sample new input (dict or list of dicts)
new_sample = {
    'RowNumber': 1,
    'CustomerId': 15634602,
    'Surname': 'Hargrave',
    'CreditScore': 619,
    'Geography': 'France',
    'Gender': 'Female',
    'Age': 42,
    'Tenure': 2,
    'Balance': 0.00,
    'NumOfProducts': 1,
    'HasCrCard': 1,
    'EstimatedSalary': 101348.88,
    'Exited':1
}

new_df = pd.DataFrame([new_sample])

# Apply the saved label encoders
for col, le in label_encoders.items():
    if col in new_df:
        new_df[col] = le.transform(new_df[col])

# Predict
prediction = best_dt_classifier.predict(new_df)
print("Prediction:", prediction[0])
