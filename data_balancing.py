import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

def resample_training_data(X_train, y_train, y_embed_train, sampling_strategy='auto', random_state=42):
    """
    Resample training data to address class imbalance.
    """
    X_columns = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
    y_embed_columns = y_embed_train.columns.tolist() if hasattr(y_embed_train, 'columns') else None
    y_name = y_train.name if hasattr(y_train, 'name') else 'depressed'

    # Convert to numpy
    X = np.array(X_train) if hasattr(X_train, 'values') else X_train
    y = np.array(y_train).ravel() if hasattr(y_train, 'values') else np.array(y_train).ravel()
    y_embed = np.array(y_embed_train) if hasattr(y_embed_train, 'values') else y_embed_train

    # Combine X and and y_embed for resampling
    n_features = X.shape[1]
    X_combined = np.hstack([X, y_embed])

    # Apply SMOTE
    sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    result = sampler.fit_resample(X_combined, y)
    (X_combined_resampled, y_resampled) = result[0], result[1]

    X_resampled = X_combined_resampled[:, :n_features]
    y_embed_resampled = X_combined_resampled[:, n_features:]
    # Round embeddings to discrete values (0, 1, 2, 3)
    y_embed_resampled = np.clip(np.round(y_embed_resampled), 0, 3).astype(int)

    # Calculate y values based on DSM-V criteria
    core_criterion = (y_embed_resampled[:, 0] >= 2) | (y_embed_resampled[:, 1] >= 2)
    symptom_count = (y_embed_resampled >= 2).sum(axis=1)
    
    depression_criteria = core_criterion & (symptom_count >= 5)
    y_resampled = depression_criteria.astype(int)
    
    X_resampled = pd.DataFrame(X_resampled, columns=X_columns)
    y_resampled = pd.Series(y_resampled, name=y_name)
    y_embed_resampled = pd.DataFrame(y_embed_resampled, columns=y_embed_columns)

    return X_resampled, y_resampled, y_embed_resampled