import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, classification_report, mean_squared_error

###########################################
# Logistic Binary Regression
###########################################
def train_binary_logistic(n_iterations, random_state, X_train, X_test, y_train, y_test):
    '''
    Trains a Logistic Regressor that predicts binary labels, and prints the performance.
    '''

    # Train
    clf = LogisticRegression(max_iter=n_iterations, class_weight='balanced', random_state=random_state)
    clf.fit(X_train, y_train)

    # Test
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n\n", "#"*40, "Binary Logistic Regression ", "#"*40)
    print("F1 score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


###########################################
# Multitarget Logistic Regressor
###########################################
def train_multitarget_logistic(n_iterations, random_state, X_train, X_test, y_train, y_test, y_embed_train, y_embed_test):
    """
    Trains a logistic regression model for each DPQ embedding item individually,
    then computes DSM-V binary labels from predicted embeddings and calculates performance on those binary labels.
    """

    target_embed_cols = ['DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090']

    # Train a logistic regression for each embedding item
    y_pred_embed = np.zeros_like(y_embed_test.to_numpy())
    for i, col in enumerate(target_embed_cols):
        clf = LogisticRegression(max_iter=n_iterations, class_weight='balanced', random_state=random_state)
        clf.fit(X_train, y_embed_train[col])
        y_pred_embed[:, i] = clf.predict(X_test)

    # Evaluate embedding prediction
    print("\n\n", "#"*40, "Multitarget Logistic Regression", "#"*40)
    mse_per_item = mean_squared_error(y_embed_test, y_pred_embed, multioutput='raw_values')
    print("MSE per DPQ item:", mse_per_item)
    print("Average MSE:", mse_per_item.mean())

    # Convert predicted embedding to DSM-V binary label
    y_pred_df = pd.DataFrame(y_pred_embed, columns=target_embed_cols)
    y_pred_df = np.clip(np.round(y_pred_df), 0, 3).astype(int)

    # Apply DSM-V criteria safely
    pred_depressed = (
        ((y_pred_df['DPQ010'] >= 2) | (y_pred_df['DPQ020'] >= 2)) &
        ((y_pred_df >= 2).sum(axis=1) >= 5)
    ).astype(int)

    # Evaluate against true binary labels
    print("F1 score:", f1_score(y_test, pred_depressed))
    print("ROC-AUC:", roc_auc_score(y_test, pred_depressed))
    print(classification_report(y_test, pred_depressed))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred_depressed))


###########################################
# Logistic Model with embedding in the input
###########################################
def train_logistic_with_embed_input(n_iterations, random_state, X_train, X_test, y_train, y_test, y_embed_train, y_embed_test):
    """
    Trains a logistic regression on X_train, combined with y_embed_train, as an additional baseline to compare our custom model against
    """

    # Combine X_train and y_embed_train
    X_train_combined = np.concatenate([X_train, y_embed_train.to_numpy()], axis=1)
    X_test_combined  = np.concatenate([X_test,  y_embed_test.to_numpy()], axis=1)

    # Train logistic regression
    clf = LogisticRegression(max_iter=n_iterations, class_weight='balanced', random_state=random_state)
    clf.fit(X_train_combined, y_train)

    # Test
    y_pred = clf.predict(X_test_combined)
    y_proba = clf.predict_proba(X_test_combined)[:, 1]

    print("\n\n", "#"*40, "Logistic Regression with Embedding Features", "#"*40)
    print("F1 score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
