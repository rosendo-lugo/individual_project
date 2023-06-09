import pandas as pd
import numpy as np


# Stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

'''
*------------------*
|                  |
|     MODELING     |
|                  |
*------------------*
'''

# ----------------------------------------------------------------------------------
def logistic_regression(X_tr, y_tr, X_val, y_val):
    '''
    X_tr = X_train
    y_tr = y_train
    X_val = X_validate
    y_val = y_validate
    '''

    # Create a Logistic Regression object
    logit = LogisticRegression(random_state=123)

    # Fit the model to the training data
    logit.fit(X_tr, y_tr)

    # Make predictions on the validation data
    y_pred = logit.predict(X_val)

    # Print the classification report
    print(classification_report(y_val, y_pred))

    return logit

# ----------------------------------------------------------------------------------
def knn(X_tr, y_tr, X_val, y_val, n_neighbors):
    '''
    X_tr = X_train
    y_tr = y_train
    X_val = X_validate
    y_val = y_validate
    n_neighbors = number of neighbors
    '''

    # Create a KNN object
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the model to the training data
    knn.fit(X_tr, y_tr)

    # Make predictions on the validation data
    y_pred = knn.predict(X_val)

    # Print the classification report
    print(classification_report(y_val, y_pred))

    return knn


# ----------------------------------------------------------------------------------
def random_forest(X_tr, y_tr, X_val, y_val, n_estimators):
    '''
    X_tr = X_train
    y_tr = y_train
    X_val = X_validate
    y_val = y_validate
    n_estimators = number of trees in the forest
    '''

    # Create a Random Forest object
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=123)

    # Fit the model to the training data
    rf.fit(X_tr, y_tr)

    # Make predictions on the validation data
    y_pred = rf.predict(X_val)

    # Print the classification report
    print(classification_report(y_val, y_pred))

    return rf

# ----------------------------------------------------------------------------------
def decision_tree(X_tr, y_tr, X_val, y_val, max_depth):
    '''
    X_tr = X_train
    y_tr = y_train
    X_val = X_validate
    y_val = y_validate
    max_depth = maximum depth of the tree
    '''

    # Create a Decision Tree object
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=123)

    # Fit the model to the training data
    dt.fit(X_tr, y_tr)

    # Make predictions on the validation data
    y_pred = dt.predict(X_val)

    # Print the classification report
    print(classification_report(y_val, y_pred))

    return dt

# ----------------------------------------------------------------------------------
def evaluate_models(X_tr, y_tr, X_ts, y_ts):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report

    # Define the models
    lr = LogisticRegression(random_state=123)
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=100, random_state=123)
    dt = DecisionTreeClassifier(max_depth=5, random_state=123)

    # Fit the models
    lr.fit(X_tr, y_tr)
    knn.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)
    dt.fit(X_tr, y_tr)

    # Make predictions on the test data
    y_pred_lr = lr.predict(X_ts)
    y_pred_knn = knn.predict(X_ts)
    y_pred_rf = rf.predict(X_ts)
    y_pred_dt = dt.predict(X_ts)

    # Print the classification reports
    print('Logistic Regression:')
    print(classification_report(y_ts, y_pred_lr))
    print('\nK-Nearest Neighbors:')
    print(classification_report(y_ts, y_pred_knn))
    print('\nRandom Forest:')
    print(classification_report(y_ts, y_pred_rf))
    print('\nDecision Tree:')
    print(classification_report(y_ts, y_pred_dt))