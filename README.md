Titanic Survival Prediction

This project builds a machine learning model to predict whether a passenger survived the Titanic disaster.
The dataset includes features like age, gender, ticket class, fare, and cabin information. 
The model uses a Random Forest classifier to achieve reliable classification results.

1.Dataset

Source: Kaggle Titanic Dataset

Target Variable: Survived (0 = No, 1 = Yes)

2. Preprocessing Steps

Filled missing values in Age, Embarked, and Fare

Dropped irrelevant columns: PassengerId, Name, and Ticket

Transformed Cabin into a binary indicator (has cabin info or not)

Encoded categorical variables (Sex, Embarked)

Normalized Age and Fare using StandardScaler

3. Model

Algorithm: Random Forest Classifier

Train/Test Split: 80/20

4. Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

5.Results 
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0

6.Output Files

outputs/titanic_rf_model.pkl: Trained model

outputs/scaler.pkl: Scaler for numerical features

7. Author
 Darshan Pareek
