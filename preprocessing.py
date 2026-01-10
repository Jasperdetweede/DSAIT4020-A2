import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

###################################
# DEPRESSION 
###################################

def preprocess_depression_data(dataset: pd.DataFrame, random_state: int):
    '''
    Preprocesses the depression dataset.
    
    :param dataset: DataFrame containing the raw combined depression dataset, containing targets and SEQN
    :param random_state: Random state for train/test split
    :return: for train and test sets: X, y, y_embed
    '''

    # Load train/test split
    X_train, X_test, y_train, y_test, y_embed_train, y_embed_test = split_depression_data(dataset, random_state)

    # 2. Drop columns with >50% missing (DECIDE USING TRAIN ONLY)
    cols_before = X_train.shape[1]
    cols_to_drop = X_train.columns[X_train.isnull().mean() > 0.5]

    X_train = X_train.drop(columns=cols_to_drop)
    X_test  = X_test.drop(columns=cols_to_drop)

    print(
        f"Dropped {len(cols_to_drop)} columns with >50% missing. "
        f"Remaining: {X_train.shape[1]} columns (was {cols_before})"
    )

    # 3. Drop rows with >50% missing
    rows_before = X_train.shape[0]
    rows_to_drop = X_train.index[X_train.isnull().mean(axis=1) > 0.5]

    X_train = X_train.drop(index=rows_to_drop)
    y_train = y_train.loc[X_train.index]
    y_embed_train = y_embed_train.loc[X_train.index]

    print(
        f"Dropped {len(rows_to_drop)} training rows with >50% missing. "
        f"Remaining: {X_train.shape[0]} rows (was {rows_before})"
    )

    # 4. Identify categorical vs numeric
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

    print(f"Categorical features: {categorical_cols}")
    print(f"Numeric features: {numeric_cols}")

    # 5. Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_cols),

            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ]
    )

    preprocessor.fit(X_train)

    # 6. Transform train and test
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed  = preprocessor.transform(X_test)

    print(f"Preprocessed train shape: {X_train_preprocessed.shape}")
    print(f"Preprocessed test shape:  {X_test_preprocessed.shape}")

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, y_embed_train, y_embed_test


def split_depression_data(dataset, random_state):
    '''
    Splits the depression dataset.
    
    :param dataset: DataFrame containing the raw combined depression dataset, containing targets and SEQN
    :param random_state: Random state for train/test split
    :return: for train and test sets: X, y_binary, y_embed
    '''

    # Read dataset from csv
    target_embed_cols = ['DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090']

    # Add binary target column based on DSM-V criteria
    depression_criteria = (
        (dataset['DPQ010'].isin([2, 3]) | dataset['DPQ020'].isin([2, 3])) &     # Little interest in doing things OR feeling down more than half the days
        (dataset[target_embed_cols].isin([2, 3]).sum(axis=1) >= 5)              # At least 5 symptoms present more than half the days 
    )

    dataset['depressed'] = (depression_criteria).astype(int)

    # Get features from the depression file, so we can drop them in X
    features_from_depression_file = list(pd.read_sas('raw_data/targets/DPQ_L_Target_Depression.xpt', format='xport').drop(columns='SEQN').columns)

    # Define sets
    X = dataset.drop(columns=features_from_depression_file).drop(columns=['SEQN', 'depressed'])
    y_embed = dataset[target_embed_cols]
    y_binary = dataset['depressed']

    # Assert correct shape and absence of SEQN column in features and targets
    assert X.shape[0] == y_embed.shape[0], "Feature and target embedding row counts do not match"
    assert X.shape[0] == y_binary.shape[0], "Feature and target binary row counts do not match"
    assert X.columns.__contains__("SEQN") == False, "Feature set should not contain SEQN column"
    assert y_embed.columns.__contains__("SEQN") == False, "Target embedding set should not contain SEQN column"

    # Split into train and test sets 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_binary,
        test_size=0.2,        
        random_state=random_state,   
        stratify=y_binary       # preserve class balance
    )

    y_embed_train = y_embed.loc[X_train.index]
    y_embed_test  = y_embed.loc[X_test.index]

    return X_train, X_test, y_train, y_test, y_embed_train, y_embed_test



###################################
# INSOMNIA 
###################################

def preprocess_insomnia_data(dataset: pd.DataFrame) -> pd.DataFrame:
    ...


