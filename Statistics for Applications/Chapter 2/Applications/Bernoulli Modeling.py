import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic_df = pd.read_csv(url)

# Data preprocessing
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
titanic_df['Pclass'] = titanic_df['Pclass'].astype('category')
titanic_df = pd.get_dummies(titanic_df, columns=['Pclass'], drop_first=True)
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Select features and target
X = titanic_df[['Sex', 'Age', 'Pclass_2', 'Pclass_3']]
y = titanic_df['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit logistic regression (which assumes a Bernoulli distribution for the outcome)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
print("\nModel Coefficients:")
print(coefficients)

# Calculate survival probabilities for different passenger profiles
profiles = pd.DataFrame({
    'Sex': [0, 1, 0, 1],
    'Age': [30, 30, 50, 50],
    'Pclass_2': [0, 0, 1, 1],
    'Pclass_3': [1, 1, 0, 0]
})

survival_probs = model.predict_proba(profiles)[:, 1]
profiles['Survival Probability'] = survival_probs

print("\nSurvival Probabilities for Different Passenger Profiles:")
print(profiles)

# Visualize the results
plt.figure(figsize=(10, 6))
sns.heatmap(profiles.iloc[:, :-1], annot=True, cmap='YlGnBu')
plt.title('Passenger Profiles')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=profiles.index, y='Survival Probability', data=profiles)
plt.title('Survival Probabilities for Different Passenger Profiles')
plt.xlabel('Profile Index')
plt.ylabel('Probability of Survival')
plt.show()

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients)
plt.title('Feature Importances')
plt.xlabel('Coefficient Value')
plt.show()