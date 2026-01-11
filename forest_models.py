import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, classification_report, mean_squared_error

###########################################
# Binary Forest
###########################################
def train_binary_forest(n_estimators, random_state, X_train, X_test, y_train, y_test):
    '''
    Trains a random forest that predicts binary labels, and prints the performance.
    '''

    # Train
    rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rfc.fit(X_train, y_train)

    # Test
    y_pred = rfc.predict(X_test)
    y_proba = rfc.predict_proba(X_test)[:, 1]

    print("\n\n", "#"*40, "binary Forest ", "#"*40)
    print("F1 score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    
    
###########################################
# Multitarget Forest Regressor
###########################################
def train_multitarget_forest(n_estimators, random_state, X_train, X_test, y_train, y_test, y_embed_train, y_embed_test):
    '''
    Trains a random forest that tries to predict each embedding variable individually.
    '''

    # Train
    rfr = RandomForestRegressor( n_estimators, random_state=random_state, n_jobs=-1)
    rfr.fit(X_train, y_embed_train)

    # Test
    print("\n\n", "#"*40, "Multitarget Forest ", "#"*40)
    y_pred_embed = rfr.predict(X_test)
    mse_per_item = mean_squared_error(y_embed_test, y_pred_embed, multioutput='raw_values')
    print("MSE per DPQ item:", mse_per_item)
    print("Average MSE:", mse_per_item.mean())

    # Convert predicted embedding to DSM-V binary label
    target_embed_cols = ['DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090']
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
# Forest with embedding in the input
###########################################
def train_forest_with_embed_input(n_estimators, random_state, X_train, X_test, y_train, y_test, y_embed_train, y_embed_test):
    '''
    Trains a random forest on X_train, combined with y_embed_train, as an additional baseline to compare our custom model against.
    '''

    # Combine features
    X_train_combined = np.concatenate([X_train, y_embed_train.to_numpy()], axis=1)
    X_test_combined  = np.concatenate([X_test, y_embed_test.to_numpy()], axis=1)

    rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rfc.fit(X_train_combined, y_train)

    # Test
    y_pred = rfc.predict(X_test_combined)
    y_proba = rfc.predict_proba(X_test_combined)[:, 1]

    print("\n\n", "#"*40, "Forest with embedding as faetures ", "#"*40)
    print("F1 score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))