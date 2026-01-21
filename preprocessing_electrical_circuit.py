import pandas as pd
import numpy as np
from electrical_circuit import ElectricalCircuit

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

###################################
# Full pipeline
###################################

def gen_and_preprocess_ec_data( sample_count: int, test_split: float, random_state: int ):
	'''
	Takes a dataframe and splits it into the training set, the corresponding target embeddings and calculates a binary label.
	Importantly there is no need for cleaning as the data is fully generated

	:param sample_count: Number of different circuits to generate
	:param test_split: Percentage of data to use as training dataset
	:param random_state: Random state for train/test split
	:return: for train and test sets: X, y, e
	'''

	# Load train/test split
	X_train, X_test, y_train, y_test, e_train, e_test = gen_and_split_ec_data( sample_count, test_split, random_state )

	# Preprocess the features
	X_train_preprocessed, X_test_preprocessed = preprocess_ec_data(X_train, X_test)

	return X_train_preprocessed, X_test_preprocessed, y_train, y_test,  e_train, e_test


###################################
# Data Splitting
###################################

def gen_and_split_ec_data( sample_count: int, test_split: float, random_state: int ):
	'''
	Generates and Splits an electrical circuit dataset and adds the binary labelling.

	:param sample_count: Number of different circuits to generate
	:param test_split: Percentage of data to use as training dataset
	:param random_state: Random state for train/test split
	:return: for train and test sets: X, e, y
	'''

	ec = ElectricalCircuit()
	X, y = ec.gen_random_samples( sample_count = sample_count, random_state=random_state )
	e = ec.get_embedding()

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_split,
		random_state=random_state
	)

	e_train = e.loc[X_train.index]
	e_test  = e.loc[X_test.index]

	return X_train, X_test, y_train, y_test, e_train, e_test


###################################
# Data Preprocessing
###################################

def preprocess_ec_data(X_train, X_test):
	'''
	Basic pipeline. All features are numeric.

	:param X_train: dataframe training set
	:param X_test: dataframe test set
	'''

	preprocessing_pipeline = ColumnTransformer(transformers=[
		('num', Pipeline([
			('scaler', StandardScaler())
		]), numerical_cols)
	])

	# Fit pipeline to training data
	preprocessing_pipeline.fit(X_train)

	# Transform training and testing data in the fitted pipeline
	X_train_preprocessed = preprocessing_pipeline.transform(X_train)
	X_test_preprocessed = preprocessing_pipeline.transform(X_test)

	return X_train_preprocessed, X_test_preprocessed
