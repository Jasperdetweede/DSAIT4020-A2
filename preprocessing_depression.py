import pandas as pd
import numpy as np
from scipy import sparse as sci

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

###################################
# Full pipeline
###################################

def clean_and_preprocess_depression_data(dataset: pd.DataFrame, raw_data_folder: str, test_split: float, random_state: int, miss_val_threshold: float):
    '''
    Takes a dataframe and splits it into the training set, the corresponding target embeddings and calculates a binary label. 

    :param dataset: DataFrame containing the raw combined depression dataset, containing targets and SEQN
    :param random_state: Random state for train/test split
    :return: for train and test sets: X, y, y_embed
    '''
    
    # Load train/test split
    X_train, X_test, y_train, y_test, y_embed_train, y_embed_test = split_depression_data(dataset, raw_data_folder, random_state, test_split)

    # Clean features 
    X_train_cleaned, X_test_cleaned = clean_depression_data(X_train, X_test, miss_val_threshold)

    # Preprocess the features
    X_train_preprocessed, X_test_preprocessed = preprocess_depression_data(X_train_cleaned, X_test_cleaned)

    target_embed_cols = ['DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090']

    # Drop rows where any target column has value 7 or 9
    valid_train_indices = y_embed_train[~y_embed_train[target_embed_cols].isin([7, 9]).any(axis=1)].index
    valid_test_indices  = y_embed_test[~y_embed_test[target_embed_cols].isin([7, 9]).any(axis=1)].index

    assert not isinstance(X_train_preprocessed, sci.spmatrix), \
       "X_train_preprocessed is still a sparse matrix! Convert to dense before indexing."
    assert not isinstance(X_test_preprocessed, sci.spmatrix), \
        "X_test_preprocessed is still a sparse matrix! Convert to dense before indexing."

    # Filter preprocessed features and targets
    X_train_preprocessed = X_train_preprocessed[np.isin(X_train.index, valid_train_indices)]
    y_train = y_train.loc[valid_train_indices]
    y_embed_train = y_embed_train.loc[valid_train_indices]

    X_test_preprocessed = X_test_preprocessed[np.isin(X_test.index, valid_test_indices)]
    y_test = y_test.loc[valid_test_indices]
    y_embed_test = y_embed_test.loc[valid_test_indices]

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, y_embed_train, y_embed_test


###################################
# Data Splitting
###################################

def split_depression_data(dataset, raw_data_folder, random_state, test_split):
    '''
    Splits the depression dataset and adds the binary labelling. 
    
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
    features_from_depression_file = list(pd.read_sas(f'{raw_data_folder}/targets/DPQ_L_Target_Depression.xpt', format='xport').drop(columns='SEQN').columns)

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
        test_size=test_split,        
        random_state=random_state,   
        stratify=y_binary       # preserve class balance
    )

    y_embed_train = y_embed.loc[X_train.index]
    y_embed_test  = y_embed.loc[X_test.index]

    return X_train, X_test, y_train, y_test, y_embed_train, y_embed_test


###################################
# Data Cleaning
###################################

def clean_depression_data(X_train: pd.DataFrame, X_test: pd.DataFrame, miss_val_threshold: float):
    '''
    :param dataset: DataFrame containing the raw combined depression dataset, containing targets and SEQN
    :param random_state: Random state for train/test split
    :return: for train and test sets: X, y, y_embed
    '''

    # 1. drop id column, unnecessary flag column and target leakage columns
    bad_columns = ['SEQN', 'SMAQUEX2', 'FN140', 'FN150', 'FN510', 'FN520', 'FNQ530', 'FNQ540']
    X_train.drop(columns=bad_columns, inplace=True, errors='ignore')
    X_test.drop(columns=bad_columns, inplace=True, errors='ignore')

    # 2. drop columns with more than 50% missing values in train set, and drop the same columns in test
    X_train, dropped_columns = drop_high_missing_columns(X_train, threshold_fraction=miss_val_threshold)
    X_test = X_test.drop(columns=dropped_columns)
    
    # 3. replace NHANES special codes with NaN in train and test set
    X_train = replace_nhanes_special_codes(X_train)
    X_test = replace_nhanes_special_codes(X_test)

    # 4. Remap values for train and test 
    binary_cols = identify_binary_columns(X_train)
    ordinal_cols, nominal_cols = get_known_categorical_columns()

    X_train = convert_time_columns(X_train)
    X_train = remap_ordinal_features(X_train)
    X_train = convert_binary_columns(X_train, binary_cols)
    X_train = shift_to_zero_indexed(X_train, ordinal_cols)

    X_test = convert_time_columns(X_test)
    X_test = remap_ordinal_features(X_test)
    X_test = convert_binary_columns(X_test, binary_cols)
    X_test = shift_to_zero_indexed(X_test, ordinal_cols)

    # 5. Combine features in tran and test set
    X_train = calculate_bmi(X_train)
    X_train = calculate_total_alcohol_consumption(X_train)

    X_test = calculate_bmi(X_test)
    X_test = calculate_total_alcohol_consumption(X_test)

    # Identify column types and print metrics
    ordinal_cols, nominal_cols, binary_cols, numerical_cols, object_cols = identify_column_types(X_train)

    print(f"Ordinal columns: {len(ordinal_cols)}")
    print(f"Nominal columns: {len(nominal_cols)}")
    print(f"Binary columns: {len(binary_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Object columns (excluded): {len(object_cols)}")
    print(f"Total columns identified: {len(ordinal_cols) + len(nominal_cols) + len(binary_cols) + len(numerical_cols) + len(object_cols)}")

    return X_train, X_test

def replace_nhanes_special_codes(df):
    """
    Replace NHANES special codes with NaN.
    - 7, 77, 777, 7777, 77777 = Refused
    - 9, 99, 999, 9999, 99999, 55555 = Don't know / Missing
    """

    exclude_cols = ['SEQN', 'ALQ121', 'ALQ130', 'ALQ142', 'ALQ270', 'ALQ280', 'ALQ170', 'HOD051', 'PAD790Q',
                    'PAD800', 'PAD810Q', 'PAD680', 'RHQ010', 'SLD012', 'SLD013', 'WHD010', 'WHD020', 'WHD050']

    # Define special codes to replace
    refused_codes = [7, 77, 777, 7777, 77777]
    dont_know_codes = [9, 99, 999, 9999, 99999, 55555]
    special_codes = refused_codes + dont_know_codes
    
    replaced_count = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in exclude_cols:
            continue
        # Replace standard special codes
        mask = df[col].isin(special_codes)
        replaced_count += mask.sum()
        df.loc[mask, col] = np.nan
    
    print(f"Replaced {replaced_count} special code values with NaN")
    return df

def drop_high_missing_columns(df, threshold_fraction: float = 0.5):
    """Drop columns with missing percentage above threshold."""
    threshold = threshold_fraction * 100
    missing_pct = df.isna().sum() / len(df) * 100
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} columns with >{threshold}% missing values")
    print(f"Shape after dropping high-missing columns: {df.shape}")
    return df, cols_to_drop

def identify_binary_columns(df, exclude_cols=[]):
    """Identify binary columns (values 1 and 2 only)."""
    binary_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in exclude_cols:
            continue
        unique_vals = set(df[col].dropna().unique())
        if unique_vals and (unique_vals.issubset({1, 2, 1.0, 2.0}) or unique_vals.issubset({0, 1, 0.0, 1.0})):
            binary_cols.append(col)
    return binary_cols

def get_known_categorical_columns():
    """Identify known categorical columns: ordinal, nominal."""
    # Known ordinal and nominal columns based on NHANES codebooks
    ordinal_cols = [
        'HUQ010',
        'FNQ410', 'FNQ430', 'FNQ440', 'FNQ450', 'FNQ460', 
        'FNQ470', 'FNQ480', 'FNQ490',
        'FNQ510', 'FNQ520',
        'FSD032A', 'FSD032B', 'FSD032C', 'FSDAD',
        'ALQ142',
        'OHQ845', 'OHQ620', 'OHQ630', 'OHQ640', 'OHQ660', 'OHQ670', 'OHQ680',
        'DIQ010', 'INDFMMPC', 'KIQ005', 'SMD460',
    ]
    
    nominal_cols = [
        'HUQ042', 'OCD150', 'HUQ030',
    ]

    return ordinal_cols, nominal_cols

def convert_time_columns(df):
    """Convert time strings (HH:MM) to decimal hours."""
    time_cols = ['SLQ300', 'SLQ310', 'SLQ320', 'SLQ330']
    
    def time_to_hours(time_str):
        if pd.isna(time_str):
            return np.nan
        try:
            if isinstance(time_str, str) and time_str.startswith("b'") and time_str.endswith("'"):
                time_str = time_str[2:-1]
            h, m = time_str.split(':')
            return int(h) + int(m) / 60
        except:
            return np.nan
    
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].apply(time_to_hours)
    return df

def calculate_bmi(df):
    """Calculate BMI from height (WHD010) in inches and weight (WHD020) in pounds."""
    # Calculate BMI from height and weight
    # WHD010 is in inches, WHD020 is in pounds
    df['BMI'] = (df['WHD020'] / (df['WHD010'] ** 2)) * 703
    df.drop(columns=['WHD010', 'WHD020'], inplace=True)
    return df

def calculate_total_alcohol_consumption(df):
    """Calculate total annual alcohol consumption estimate."""
    # Convert ALQ121 to days per year
    alq121_to_days = {
        0: 0,      # Never
        1: 365,    # Every day (original code 1)
        2: 286,    # Nearly every day
        3: 182,    # 3-4 times/week
        4: 104,    # 2 times/week  
        5: 52,     # Once/week
        6: 30,     # 2-3 times/month
        7: 12,     # Once/month
        8: 9,      # 7-11 times/year
        9: 4.5,    # 3-6 times/year
        10: 1.5    # 1-2 times/year
    }
    df['alcohol_days_per_year'] = df['ALQ121'].map(alq121_to_days)

    # Total alcohol consumption estimate
    df['annual_drinks'] = df['alcohol_days_per_year'] * df['ALQ130']
    
    df.drop(columns=['ALQ121', 'ALQ130'], inplace=True)
    return df

def remap_ordinal_features(df):
    # for some ordinal features, higher value = less frequent/severe => should be reversed
    remappings = {
        # FNQ510: 1=daily → 5=never (should be: higher = more frequent)
        'FNQ510': {1: 4, 2: 3, 3: 2, 4: 1, 5: 0},  # daily→4, weekly→3, monthly→2, few times→1, never→0
        
        # OHQ620-OHQ680: 1=very often → 5=never (should be: higher = more problems)
        'OHQ620': {1: 4, 2: 3, 3: 2, 4: 1, 5: 0},
        'OHQ630': {1: 4, 2: 3, 3: 2, 4: 1, 5: 0},
        'OHQ640': {1: 4, 2: 3, 3: 2, 4: 1, 5: 0},
        'OHQ660': {1: 4, 2: 3, 3: 2, 4: 1, 5: 0},
        'OHQ670': {1: 4, 2: 3, 3: 2, 4: 1, 5: 0},
        'OHQ680': {1: 4, 2: 3, 3: 2, 4: 1, 5: 0},
        
        # FSD032A-C: 1=often true → 3=never true (should be: higher = more food insecurity)
        'FSD032A': {1: 2, 2: 1, 3: 0},
        'FSD032B': {1: 2, 2: 1, 3: 0},
        'FSD032C': {1: 2, 2: 1, 3: 0},

        # ALQ121, ALQ142: 0=Never, 1=Every day → 10=1-2 times a year (should be: higher = more frequent)
        # 'ALQ121': {0: 0, 10: 1, 9: 2, 8: 3, 7: 4, 6: 5, 5: 6, 4: 7, 3: 8, 2: 9, 1: 10},
        'ALQ142': {0: 0, 10: 1, 9: 2, 8: 3, 7: 4, 6: 5, 5: 6, 4: 7, 3: 8, 2: 9, 1: 10},

        # DIQ010: 1=Yes, 2=No, 3=Borderline
        # Correct order: No(0) → Borderline(1) → Yes(2)
        'DIQ010': {2: 0, 3: 1, 1: 2},  
        
        # FNQ520: 1=a little, 2=a lot, 3=somewhere in between
        # Correct order: little(0) → between(1) → a lot(2)
        'FNQ520': {1: 0, 3: 1, 2: 2},
    }

    for col, mapping in remappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            
    return df

def convert_binary_columns(df, binary_cols):
    """Convert binary columns from (1,2) to (1,0)."""
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].replace({2: 0, 2.0: 0})
    return df

def shift_to_zero_indexed(df, columns):
    """Shift ordinal columns starting at 1 to start at 0."""
    for col in columns:
        if col in df.columns:
            min_val = df[col].dropna().min()
            if min_val >= 1:
                df[col] = df[col] - min_val
    return df

def identify_column_types(df, ignore_columns=[]):
    """
    Separate columns into ordinal, nominal, binary, object (categorical) and numerical types.
    """
    exclude_cols = set(ignore_columns)
    object_cols = df.select_dtypes(include='object').columns.tolist()
    exclude_cols.update(object_cols)

    # Get known ordinal and nominal columns
    ordinal_cols, nominal_cols = get_known_categorical_columns()

    ordinal_cols = [col for col in ordinal_cols if col in df.columns]
    nominal_cols = [col for col in nominal_cols if col in df.columns]

    # Identify binary columns (Yes=1, No=2)
    binary_cols = identify_binary_columns(df, exclude_cols)
    
    # Combine all known categorical columns
    known_categorical = set(ordinal_cols + nominal_cols + binary_cols)

    # Build numerical list
    numerical_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in known_categorical and col not in exclude_cols:
            numerical_cols.append(col)
    
    return ordinal_cols, nominal_cols, binary_cols, numerical_cols, object_cols


###################################
# Data Preprocessing
###################################

def preprocess_depression_data(X_train, X_test):
    '''
    Basic pipeline. Imputes and scales, and encodes nominal features. 
    
    :param X_train: dataframe training set
    :param X_test: dataframe test set
    '''

    ordinal_cols, nominal_cols, binary_cols, numerical_cols, object_cols = identify_column_types(X_train)

    preprocessing_pipeline = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        
        ('ord', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ]), ordinal_cols),
        
        ('nom', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ]), nominal_cols),
        
        ('bin', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ]), binary_cols),
    ])

    # Fit pipeline to training data
    preprocessing_pipeline.fit(X_train)

    # Transform training and testing data in the fitted pipeline
    X_train_preprocessed = preprocessing_pipeline.transform(X_train)
    X_test_preprocessed = preprocessing_pipeline.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed

