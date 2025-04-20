# TitanicSurvivalprediction
# Titanic Survival Prediction

## Project Overview
This project aims to predict whether a passenger survived the Titanic disaster using machine learning. The dataset includes features like age, gender, ticket class, fare, and cabin information. The goal is to achieve high accuracy in predicting passenger survival.

## Dataset
The dataset used in this project is the Titanic dataset, which includes the following columns:
- `PassengerId`: Unique identifier for each passenger.
- `Survived`: Survival (0 = No, 1 = Yes).
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- `Name`: Name of the passenger.
- `Sex`: Gender of the passenger.
- `Age`: Age in years.
- `SibSp`: Number of siblings/spouses aboard.
- `Parch`: Number of parents/children aboard.
- `Ticket`: Ticket number.
- `Fare`: Passenger fare.
- `Cabin`: Cabin number.
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Data Preprocessing

### 1. Import Libraries
Import necessary libraries for data manipulation, preprocessing, model training, and evaluation.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

2. Load the Dataset
Load the Titanic dataset from a CSV file.

data = pd.read_csv("titanic.csv")
3. Drop Unnecessary Columns
Drop columns that are not useful for the model.
data.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

4. Handle Missing Values
Fill missing values in Age, Embarked, and Fare with the median or mode.
Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
5. Feature Engineering
Create new features that might be useful for the model.

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = np.where(data['FamilySize'] == 1, 1, 0)
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
data['Title'] = data['Title'].map(title_mapping)
data['Title'] = data['Title'].fillna(0)
6. Encode Categorical Variables
Convert categorical variables to numeric values.

label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])
7. Drop Name Column
Drop the Name column as it is no longer needed.

data.drop(['Name'], axis=1, inplace=True)
8. Separate Features and Target Variable
Separate the features (X) and the target variable (y).
X = data.drop('Survived', axis=1)
y = data['Survived']
9. Train-Test Split
Split the data into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
10. Normalize Numerical Data
Normalize numerical data using StandardScaler.

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Model Selection and Hyperparameter Tuning
11. Hyperparameter Tuning using GridSearchCV
Use GridSearchCV to find the best hyperparameters for the SVC model.

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

model = SVC()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
12. Best Parameters and Best Score
Print the best parameters and best cross-validation score.

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)
13. Train the Best Model
Train the best model found by GridSearchCV.

best_model = grid_search.best_estimator_
14. Make Predictions
Make predictions on the test set.

y_pred = best_model.predict(X_test)
15. Evaluate Model Performance
Evaluate the model's performance using accuracy and a classification report.

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
Conclusion
This project demonstrates the process of predicting Titanic passenger survival using machine learning. By following the steps outlined in this README, you can preprocess the data, engineer features, select a model, tune hyperparameters, and evaluate the model's performance. The goal is to achieve high accuracy in predicting passenger survival.
Usage
To run the model, clone this repository and execute the script.
bash
Copy
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
python titanic_model.py
Dependencies
pandas
numpy
scikit-learn
License
This project is licensed under the MIT License - see the LICENSE file for details.
Copy

### Explanation of Each Step

1. **Import Libraries**: Import necessary libraries for data manipulation, preprocessing, model training, and evaluation.
2. **Load the Dataset**: Load the Titanic dataset from a CSV file.
3. **Drop Unnecessary Columns**: Remove columns that are not useful for the model.
4. **Handle Missing Values**: Fill missing values in `Age`, `Embarked`, and `Fare` with the median or mode.
5. **Feature Engineering**: Create new features like `FamilySize`, `IsAlone`, and `Title` to provide additional context.
6. **Encode Categorical Variables**: Convert categorical variables to numeric values using `LabelEncoder`.
7. **Drop Name Column**: Remove the `Name` column after extracting necessary information.
8. **Separate Features and Target Variable**: Separate the features (`X`) and the target variable (`y`).
9. **Train-Test Split**: Split the data into training and testing sets.
10. **Normalize Numerical Data**: Normalize numerical data using `StandardScaler`.
11. **Hyperparameter Tuning**: Use `GridSearchCV` to find the best hyperparameters for the `SVC` model.
12. **Best Parameters and Best Score**: Print the best parameters and best cross-validation score.
13. **Train the Best Model**: Train the best model found by `GridSearchCV`.
14. **Make Predictions**: Make predictions on the test set.
15. **Evaluate Model Performance**: Evaluate the model's performance using 
