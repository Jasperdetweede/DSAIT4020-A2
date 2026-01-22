import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, classification_report, mean_squared_error

###########################################
# Binary GaussianNB
###########################################
def train_binary_bayes(X_train, X_test, y_train, y_test):
    '''
    Trains a Gaussian Naive Bayes classifier on binary labels and prints performance.
    '''

    # Train
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Test
    y_pred = gnb.predict(X_test)
    y_proba = gnb.predict_proba(X_test)[:, 1]

    print("\n\n", "#"*40, "Binary GaussianNB Model", "#"*40)
    print("F1 score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


###########################################
# Multitarget regression using Bayesian Ridge
###########################################
def train_multitarget_bayes(X_train, X_test, y_train, y_test, y_embed_train, y_embed_test):
    '''
    Trains a Bayesian Ridge regressor to predict each embedding variable individually.
    '''

    # Train
    br = MultiOutputRegressor(BayesianRidge())
    br.fit(X_train, y_embed_train)

    # Test
    print("\n\n", "#"*40, "Multitarget Bayesian Ridge Model", "#"*40)
    y_pred_embed = br.predict(X_test)
    mse_per_item = mean_squared_error(y_embed_test, y_pred_embed, multioutput='raw_values')
    print("MSE per DPQ item:", mse_per_item)
    print("Average MSE:", mse_per_item.mean())

    # Convert predicted embedding to DSM-V binary label
    target_embed_cols = ['DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090']
    y_pred_embed = np.asarray(y_pred_embed)
    y_pred_df = pd.DataFrame(y_pred_embed, columns=target_embed_cols)
    y_pred_df = np.clip(np.round(y_pred_df), 0, 3).astype(int)

    # Apply DSM-V criteria
    pred_depressed = (
        ((y_pred_df['DPQ010'] >= 2) | (y_pred_df['DPQ020'] >= 2)) &
        ((y_pred_df >= 2).sum(axis=1) >= 5)
    ).astype(int)

    # Evaluate against binary ground truth
    print("F1 score:", f1_score(y_test, pred_depressed))
    print("ROC-AUC:", roc_auc_score(y_test, pred_depressed))
    print(classification_report(y_test, pred_depressed))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred_depressed))


###########################################
# Bayesian model with embedding in the input
###########################################
def train_bayes_with_embed(X_train, X_test, y_train, y_test, y_embed_train, y_embed_test):
    '''
    Trains a GaussianNB on X_train combined with embedding features as additional input.
    '''

    # Combine features
    X_train_combined = np.concatenate([X_train, y_embed_train.to_numpy()], axis=1)
    X_test_combined  = np.concatenate([X_test, y_embed_test.to_numpy()], axis=1)

    # Train
    gnb = GaussianNB()
    gnb.fit(X_train_combined, y_train)

    # Test
    y_pred = gnb.predict(X_test_combined)
    y_proba = gnb.predict_proba(X_test_combined)[:, 1]

    print("\n\n", "#"*40, "GaussianNB Model with Embedding Features", "#"*40)
    print("F1 score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
